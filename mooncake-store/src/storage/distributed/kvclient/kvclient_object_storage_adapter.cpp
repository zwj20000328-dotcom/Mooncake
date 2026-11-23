#include "storage/distributed/kvclient/kvclient_object_storage_adapter.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string_view>
#include <utility>
#include <vector>

#include <glog/logging.h>
#include <xxhash.h>

#include "storage/distributed/kvclient/kvclient_adapter_factory.h"

namespace mooncake {
namespace {

inline constexpr uint64_t kKvClientHashSeed = 5381;

}  // namespace

KvClientObjectStorageAdapter::KvClientObjectStorageAdapter(
    KvClientAdapterConfig config)
    : config_(std::move(config)) {}

KvClientObjectStorageAdapter::KvClientObjectStorageAdapter(
    KvClientAdapterConfig config, std::unique_ptr<kv::KvClient> client)
    : config_(std::move(config)), kv_client_(std::move(client)) {}

KvClientObjectStorageAdapter::~KvClientObjectStorageAdapter() {
    // No separate Shutdown() method: stop accepting work and let the SDK drain
    // pending tasks.
    if (kv_client_) kv_client_->Shutdown();
}

tl::expected<void, ErrorCode> KvClientObjectStorageAdapter::Init() {
    if (initialized_) return {};
    if (!config_.Validate())
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);

    // Injected client (test seam): only the adapter config is validated.
    if (kv_client_) {
        initialized_ = true;
        return {};
    }

    auto client = kv::CreateKvClient();
    if (!client) return tl::make_unexpected(ErrorCode::DFS_SERVICE_UNAVAILABLE);

    kv::KvClientConfig cfg;
    if (!kv::KvClientConfig::LoadFromFile(config_.config_path, cfg).ok())
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    cfg.keyResolution = {};  // Force GENERIC.
    // Single-entry LoadAsync/StoreAsync must use the batch protocol: the
    // non-batch LOAD/STORE path returns only a CQE status with no per-entry
    // result buffer, so a read miss would surface as CQE_CHECK_RESULT_BUFFER
    // instead of an OBJECT_NOT_FOUND entry code.
    cfg.useBatchIoForSingleKey = true;
    // Init is not idempotent; the local client is only committed to the member
    // after this succeeds, so a retry starts from a fresh client.
    if (!client->Init(cfg).ok())
        return tl::make_unexpected(ErrorCode::DFS_SERVICE_UNAVAILABLE);

    kv_client_ = std::move(client);
    initialized_ = true;
    LOG(INFO) << "KVClient adapter initialized";
    return {};
}

ErrorCode KvClientObjectStorageAdapter::ToErrorCode(const kv::Status& status,
                                                    bool is_write) {
    using kv::StatusCode;
    switch (status.code) {
        case StatusCode::OK:
            return ErrorCode::OK;
        case StatusCode::TIMEOUT:
        case StatusCode::CQE_IO_TIMEOUT:
            return ErrorCode::RPC_TIMEOUT;
        case StatusCode::IN_PROGRESS:
        case StatusCode::CANCELED:
            return ErrorCode::INTERNAL_ERROR;

        // QUERY reports a miss as ENTRY_KEY_NOT_EXIST; LOAD and BATCH_STORE
        // share one decoder and use these two codes. On a read they mean the
        // object is missing; on a write they mean the store failed, which is
        // not a miss and must not be reported as OBJECT_NOT_FOUND.
        case StatusCode::CQE_KEY_NOT_EXIST:
        case StatusCode::ENTRY_KEY_NOT_EXIST:
            return ErrorCode::OBJECT_NOT_FOUND;
        case StatusCode::ENTRY_KEY_NOT_FOUND:
        case StatusCode::ENTRY_DATA_NOT_EXIST:
            return is_write ? ErrorCode::FILE_WRITE_FAIL
                            : ErrorCode::OBJECT_NOT_FOUND;

        // NOT_FOUND is a routing/config/view error, not a missing object.
        // Mapping it to OBJECT_NOT_FOUND would let PR3 delete metadata and
        // PR2 evict live metadata on a routing failure.
        case StatusCode::NOT_FOUND:
        case StatusCode::NOT_INITIALIZED:
        case StatusCode::CONNECTION_ERROR:
        case StatusCode::IO_ERROR:
        case StatusCode::UNSUPPORTED:
        // Delete entry failures use ENTRY_DELETE_FAILED; ENTRY_KEY_NOT_EXIST
        // is produced only for QUERY.
        case StatusCode::ENTRY_DELETE_FAILED:
        // Storage-side conditions the SDK tags retryable. A single-entry LOAD
        // reports the same condition as IO_ERROR, so this keeps the one-entry
        // and batch paths consistent.
        case StatusCode::ENTRY_RETRY_ADVISED:
            return ErrorCode::DFS_SERVICE_UNAVAILABLE;

        // Explicitly non-retryable: retrying cannot succeed, so this must not
        // read as a transient service outage.
        case StatusCode::ENTRY_NO_RETRY_ADVISED:
            return ErrorCode::INTERNAL_ERROR;

        case StatusCode::INVALID_ARGUMENT:
        case StatusCode::BUFFER_NOT_REGISTERED:
        case StatusCode::BUFFER_NOT_SUPPORTED:
            return ErrorCode::INVALID_PARAMS;
        case StatusCode::CQE_OUT_OF_CREATE_SIZE:
        case StatusCode::RESOURCE_BUSY:
        case StatusCode::CQE_RESOURCE_BUSY:
            return ErrorCode::NO_AVAILABLE_HANDLE;
        default:
            return ErrorCode::INTERNAL_ERROR;
    }
}

KvClientObjectStorageAdapter::Completion
KvClientObjectStorageAdapter::waitForCompletion(kv::TaskId task_id,
                                                size_t entry_count) {
    kv::TaskResult result;
    // 0 lets the SDK use its configured defaultWaitTimeoutMs; the adapter does
    // not impose a second timeout. The SDK reports TIMEOUT via the result.
    const auto wait_status = kv_client_->Wait(task_id, 0, result);
    const auto code = wait_status.ok() ? result.status.code : wait_status.code;
    const kv::Status& task_status =
        wait_status.ok() ? result.status : wait_status;

    Completion completion;
    switch (code) {
        case kv::StatusCode::OK:
        case kv::StatusCode::PARTIAL_FAILED:
            // Only these task statuses carry meaningful per-entry results.
            completion.terminal = true;
            break;
        case kv::StatusCode::TASK_NOT_FOUND:
            // Wait removes the task even after a timeout, so an unknown task
            // does not prove that its I/O has finished. Keep it non-terminal
            // so callers retain handles and staging buffers.
            completion.terminal = false;
            completion.entryStatus.assign(
                entry_count, kv::Status::Error(kv::StatusCode::INTERNAL_ERROR,
                                               "task not found"));
            return completion;
        case kv::StatusCode::TIMEOUT:
        case kv::StatusCode::IN_PROGRESS:
        case kv::StatusCode::CANCELED:
            // The SDK may have populated entryStatus before returning the
            // timeout, but the task is not terminal; per-entry results are not
            // trustworthy. Fill with the task-level status and retain
            // resources.
            completion.terminal = false;
            completion.entryStatus.assign(entry_count, task_status);
            return completion;
        default:
            // A final task-level error. entryStatus may still hold submit-time
            // OK values, so propagate the overall error to every entry instead
            // of reading it.
            completion.terminal = true;
            completion.entryStatus.assign(entry_count, task_status);
            return completion;
    }

    if (result.entryStatus.size() != entry_count) {
        completion.entryStatus.assign(
            entry_count, kv::Status::Error(kv::StatusCode::INTERNAL_ERROR,
                                           "entry count mismatch"));
        return completion;
    }
    completion.entryStatus = std::move(result.entryStatus);
    return completion;
}

kv::BinaryKey KvClientObjectStorageAdapter::MakeCacheKey(
    std::string_view logical_key) const {
    uint64_t hash =
        XXH64(logical_key.data(), logical_key.size(), kKvClientHashSeed);
    // The protocol rejects an all-zero key; map it to the seed to keep the
    // wire key non-zero without a second constant.
    if (hash == 0) hash = kKvClientHashSeed;
    kv::BinaryKey key{};
    // Pack explicitly little-endian like the SDK's key_resolver; a memcpy of
    // the host-order uint64_t would differ on big-endian clients.
    for (size_t i = 0; i < sizeof(hash); ++i)
        key[i] = static_cast<std::byte>(hash >> (8 * i));
    return key;
}

tl::expected<uint64_t, ErrorCode> KvClientObjectStorageAdapter::RegisterBuffers(
    void* ptr, size_t size) {
    if (ptr == nullptr || size == 0) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }
    std::vector<kv::MemoryRegion> regions{
        {kv::MemoryType::HOST, reinterpret_cast<uint64_t>(ptr), size, -1, -1}};
    std::vector<kv::RegisteredMemory> registered;
    const auto status = kv_client_->RegisterRegions(regions, registered);
    if (!status.ok() || registered.size() != 1) {
        return tl::make_unexpected(ErrorCode::DFS_SERVICE_UNAVAILABLE);
    }
    return registered[0].handle;
}

void KvClientObjectStorageAdapter::UnregisterBuffers(
    const std::vector<uint64_t>& handles) {
    if (handles.empty()) return;
    // One batched call: the SDK skips invalid and unknown handles itself, and
    // batching avoids taking its memory lock once per handle. A failed
    // unregister is only logged; the SDK keeps those registrations alive and
    // there is no handle left to retry them with.
    const auto status = kv_client_->UnregisterRegions(handles);
    if (!status.ok()) {
        LOG(WARNING) << "KVClient buffer unregister failed, handles="
                     << handles.size() << ", error=" << status.message;
    }
}

tl::expected<void, ErrorCode> KvClientObjectStorageAdapter::Delete(
    const std::string& key) {
    std::array<std::string, 1> keys{key};
    auto results = DeleteBatch(keys);
    return results.empty() ? tl::make_unexpected(ErrorCode::INVALID_PARAMS)
                           : std::move(results.front());
}

ObjectStorageIoResults KvClientObjectStorageAdapter::DeleteBatch(
    std::span<const std::string> keys) {
    if (!initialized_ || !kv_client_) {
        return ObjectStorageIoResults(
            keys.size(),
            tl::make_unexpected(ErrorCode::DFS_SERVICE_UNAVAILABLE));
    }
    ObjectStorageIoResults results(keys.size());
    if (keys.empty()) return results;

    std::vector<kv::CacheKey> cache_keys;
    std::vector<size_t> indices;
    cache_keys.reserve(keys.size());
    indices.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        if (keys[i].empty()) {
            results[i] = tl::make_unexpected(ErrorCode::INVALID_PARAMS);
            continue;
        }
        cache_keys.push_back(kv::CacheKey::FromBytes(MakeCacheKey(keys[i])));
        indices.push_back(i);
    }
    if (cache_keys.empty()) return results;

    kv::TaskId task_id = kv::kInvalidTaskId;
    const auto submit = kv_client_->DeleteAsync(cache_keys, task_id);
    if (!submit.ok()) {
        // Submission failed before the SDK accepted the task; propagate the
        // overall error to every submitted entry.
        for (size_t i : indices)
            results[i] = tl::make_unexpected(ToErrorCode(submit, false));
        return results;
    }

    const auto completion = waitForCompletion(task_id, cache_keys.size());
    for (size_t j = 0; j < indices.size(); ++j) {
        results[indices[j]] = completion.entryStatus[j].ok()
                                  ? tl::expected<void, ErrorCode>{}
                                  : tl::make_unexpected(ToErrorCode(
                                        completion.entryStatus[j], false));
    }
    return results;
}

tl::expected<void, ErrorCode> KvClientObjectStorageAdapter::Put(
    const std::string& logical_key, std::span<const char> data) {
    iovec iov{const_cast<char*>(data.data()), data.size()};
    return PutV(logical_key, &iov, 1);
}

tl::expected<void, ErrorCode> KvClientObjectStorageAdapter::PutV(
    const std::string& logical_key, const iovec* iov, int iovcnt) {
    auto results = PutBatch({{logical_key, iov, iovcnt}});
    return results.empty() ? tl::make_unexpected(ErrorCode::INVALID_PARAMS)
                           : std::move(results.front());
}

std::vector<tl::expected<void, ErrorCode>>
KvClientObjectStorageAdapter::PutBatch(
    const std::vector<ObjectPutRequest>& requests) {
    std::vector<tl::expected<void, ErrorCode>> results(requests.size());
    if (!initialized_ || !kv_client_) {
        for (auto& result : results) {
            result = tl::make_unexpected(ErrorCode::DFS_SERVICE_UNAVAILABLE);
        }
        return results;
    }

    std::vector<uint64_t> handles;
    handles.reserve(requests.size());
    // Registered handles must be released even when the batch unwinds.
    try {
        std::set<kv::BinaryKey> seen;  // Same-batch duplicate keys are only a
                                       // write concern.
        std::vector<kv::KVBuffer> entries;
        std::vector<size_t> indices;
        entries.reserve(requests.size());
        indices.reserve(requests.size());
        for (size_t i = 0; i < requests.size(); ++i) {
            if (requests[i].logical_key.empty() || requests[i].iov == nullptr ||
                requests[i].iovcnt <= 0) {
                results[i] = tl::make_unexpected(ErrorCode::INVALID_PARAMS);
                continue;
            }
            const auto binary_key = MakeCacheKey(requests[i].logical_key);
            if (seen.find(binary_key) != seen.end()) {
                results[i] = tl::make_unexpected(ErrorCode::INVALID_PARAMS);
                continue;
            }
            // Sum here: an empty segment or a wrapped total would register
            // and submit the wrong range. The SDK reads one contiguous range
            // per entry, so every segment must also physically follow the
            // previous one; the running total doubles as that offset. Address
            // arithmetic uses integers because comparing pointers from
            // different allocations is undefined.
            const auto base =
                reinterpret_cast<uintptr_t>(requests[i].iov[0].iov_base);
            size_t total = 0;
            bool valid = true;
            for (int j = 0; j < requests[i].iovcnt; ++j) {
                const auto& segment = requests[i].iov[j];
                if (segment.iov_base == nullptr || segment.iov_len == 0 ||
                    segment.iov_len >
                        std::numeric_limits<size_t>::max() - total) {
                    valid = false;
                    break;
                }
                if (reinterpret_cast<uintptr_t>(segment.iov_base) !=
                    base + total) {
                    valid = false;
                    break;
                }
                total += segment.iov_len;
            }
            if (!valid || total == 0) {
                results[i] = tl::make_unexpected(ErrorCode::INVALID_PARAMS);
                continue;
            }
            auto handle = RegisterBuffers(requests[i].iov[0].iov_base, total);
            if (!handle) {
                results[i] = tl::make_unexpected(handle.error());
                continue;
            }
            // Record the handle first so a throwing allocation below still
            // releases it; only a fully accepted request claims its key.
            handles.push_back(*handle);
            seen.insert(binary_key);

            kv::KVBuffer entry;
            entry.key = kv::CacheKey::FromBytes(binary_key);
            entry.buffer.region = {
                kv::MemoryType::HOST,
                reinterpret_cast<uint64_t>(requests[i].iov[0].iov_base), total,
                -1, -1};
            entry.buffer.handle = *handle;
            entry.offset = 0;
            entry.mrKey = std::nullopt;  // SDK resolves it from the handle.
            entries.push_back(std::move(entry));
            indices.push_back(i);
        }

        if (!entries.empty()) {
            kv::TaskId task_id = kv::kInvalidTaskId;
            const auto submit = kv_client_->StoreAsync(entries, task_id);
            if (!submit.ok()) {
                for (size_t index : indices) {
                    results[index] =
                        tl::make_unexpected(ToErrorCode(submit, false));
                }
            } else {
                const auto completion =
                    waitForCompletion(task_id, entries.size());
                if (completion.entryStatus.size() != entries.size()) {
                    for (size_t index : indices) {
                        results[index] =
                            tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
                    }
                } else if (!completion.terminal) {
                    // Task-level status was fanned out to every entry.
                    const auto error =
                        ToErrorCode(completion.entryStatus[0], false);
                    for (size_t index : indices) {
                        results[index] = tl::make_unexpected(error);
                    }
                } else {
                    for (size_t j = 0; j < entries.size(); ++j) {
                        const auto& status = completion.entryStatus[j];
                        results[indices[j]] =
                            status.ok() ? tl::expected<void, ErrorCode>{}
                                        : tl::make_unexpected(
                                              ToErrorCode(status, true));
                    }
                }
            }
        }
    } catch (...) {
        UnregisterBuffers(handles);
        throw;
    }
    UnregisterBuffers(handles);
    return results;
}

tl::expected<size_t, ErrorCode> KvClientObjectStorageAdapter::Get(
    const std::string& logical_key, void* buf, size_t len) {
    auto results = GetBatch({{logical_key, buf, len}});
    return results.empty() ? tl::make_unexpected(ErrorCode::INVALID_PARAMS)
                           : std::move(results.front());
}

std::vector<tl::expected<size_t, ErrorCode>>
KvClientObjectStorageAdapter::GetBatch(
    const std::vector<ObjectGetRequest>& requests) {
    std::vector<tl::expected<size_t, ErrorCode>> results(requests.size());
    if (!initialized_ || !kv_client_) {
        for (auto& result : results) {
            result = tl::make_unexpected(ErrorCode::DFS_SERVICE_UNAVAILABLE);
        }
        return results;
    }

    std::vector<uint64_t> handles;
    handles.reserve(requests.size());
    // Registered handles must be released even when the batch unwinds.
    try {
        std::vector<kv::KVBuffer> entries;
        std::vector<size_t> indices;
        entries.reserve(requests.size());
        indices.reserve(requests.size());
        for (size_t i = 0; i < requests.size(); ++i) {
            if (requests[i].logical_key.empty() ||
                requests[i].buffer == nullptr || requests[i].size == 0) {
                results[i] = tl::make_unexpected(ErrorCode::INVALID_PARAMS);
                continue;
            }
            // The whole target range is registered; the SDK writes the object
            // straight into it.
            auto handle = RegisterBuffers(requests[i].buffer, requests[i].size);
            if (!handle) {
                results[i] = tl::make_unexpected(handle.error());
                continue;
            }
            // Record the handle first so a throwing allocation below still
            // releases it.
            handles.push_back(*handle);

            kv::KVBuffer entry;
            entry.key =
                kv::CacheKey::FromBytes(MakeCacheKey(requests[i].logical_key));
            entry.buffer.region = {
                kv::MemoryType::HOST,
                reinterpret_cast<uint64_t>(requests[i].buffer),
                requests[i].size, -1, -1};
            entry.buffer.handle = *handle;
            entry.offset = 0;
            entry.mrKey = std::nullopt;  // SDK resolves it from the handle.
            entries.push_back(std::move(entry));
            indices.push_back(i);
        }

        if (!entries.empty()) {
            kv::TaskId task_id = kv::kInvalidTaskId;
            const auto submit = kv_client_->LoadAsync(entries, task_id);
            if (!submit.ok()) {
                for (size_t index : indices) {
                    results[index] =
                        tl::make_unexpected(ToErrorCode(submit, false));
                }
            } else {
                const auto completion =
                    waitForCompletion(task_id, entries.size());
                if (completion.entryStatus.size() != entries.size()) {
                    for (size_t index : indices) {
                        results[index] =
                            tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
                    }
                } else if (!completion.terminal) {
                    // Task-level status was fanned out to every entry.
                    const auto error =
                        ToErrorCode(completion.entryStatus[0], false);
                    for (size_t index : indices) {
                        results[index] = tl::make_unexpected(error);
                    }
                } else {
                    for (size_t j = 0; j < entries.size(); ++j) {
                        const auto& status = completion.entryStatus[j];
                        const size_t index = indices[j];
                        // Report the requested length: the SDK exposes no
                        // transferred byte count, so a truncated read cannot
                        // be detected here and is left to checksum
                        // verification above.
                        results[index] =
                            status.ok()
                                ? tl::expected<size_t,
                                               ErrorCode>{requests[index].size}
                                : tl::make_unexpected(
                                      ToErrorCode(status, false));
                    }
                }
            }
        }
    } catch (...) {
        UnregisterBuffers(handles);
        throw;
    }
    UnregisterBuffers(handles);
    return results;
}

tl::expected<bool, ErrorCode> KvClientObjectStorageAdapter::Exists(
    const std::string& logical_key) {
    if (!initialized_ || !kv_client_) {
        return tl::make_unexpected(ErrorCode::DFS_SERVICE_UNAVAILABLE);
    }
    // Reject locally; an empty key never reaches the SDK.
    if (logical_key.empty()) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }

    std::vector<kv::CacheKey> keys{
        kv::CacheKey::FromBytes(MakeCacheKey(logical_key))};
    kv::TaskId task_id = kv::kInvalidTaskId;
    const auto submit = kv_client_->QueryAsync(keys, task_id);
    if (!submit.ok()) return tl::make_unexpected(ToErrorCode(submit, false));

    // A missing key leaves the task PARTIAL_FAILED, which still carries a
    // trustworthy per-entry result, so it must not be treated as a failure.
    const auto completion = waitForCompletion(task_id, keys.size());
    if (!completion.terminal) {
        return tl::make_unexpected(
            ToErrorCode(completion.entryStatus[0], false));
    }
    // Both outcomes are reported as Error statuses, so ok() cannot tell a hit
    // from a miss; branch on the code instead.
    switch (completion.entryStatus[0].code) {
        case kv::StatusCode::ENTRY_KEY_EXIST:
            return true;
        case kv::StatusCode::ENTRY_KEY_NOT_EXIST:
            return false;
        default:
            return tl::make_unexpected(
                ToErrorCode(completion.entryStatus[0], false));
    }
}

tl::expected<std::vector<KeyInfo>, ErrorCode>
KvClientObjectStorageAdapter::ListKeys() {
    return tl::make_unexpected(ErrorCode::NOT_SUPPORTED);
}

tl::expected<std::unique_ptr<ObjectStorageAdapter>, ErrorCode>
CreateKvClientObjectStorageAdapter() {
    auto config = KvClientAdapterConfig::FromEnvironment();
    if (!config) return tl::make_unexpected(config.error());
    return std::make_unique<KvClientObjectStorageAdapter>(*config);
}

}  // namespace mooncake
