#include "storage/distributed/kvclient/kvclient_object_storage_adapter.h"

#include <sys/uio.h>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <xxhash.h>

namespace mooncake {
namespace {

constexpr uint64_t kHashSeed = 5381;

kv::Status Injected(kv::StatusCode code) {
    return kv::Status::Error(code, "injected");
}

// Records every SDK interaction so tests can assert call counts and order, and
// lets them inject submit, task and per-entry failures. RegisterRegions hands
// out one monotonic handle per region, in input order.
class FakeKvClient final : public kv::KvClient {
   public:
    kv::Status Init(const kv::KvClientConfig&) override { return {}; }

    kv::Status Shutdown() override {
        ++shutdown_calls_;
        return {};
    }

    kv::Status QueryAsync(const std::vector<kv::CacheKey>& keys,
                          kv::TaskId& task_id, const kv::IoOptions&) override {
        ++query_calls_;
        query_keys_ = keys;
        return StartTask(keys.size(), task_id, query_submit_status_);
    }

    kv::Status LoadAsync(const std::vector<kv::KVBuffer>& entries,
                         kv::TaskId& task_id, const kv::IoOptions&) override {
        ++load_calls_;
        load_entries_ = entries;
        return StartTask(entries.size(), task_id, load_submit_status_);
    }

    kv::Status StoreAsync(const std::vector<kv::KVBuffer>& entries,
                          kv::TaskId& task_id, const kv::IoOptions&) override {
        ++store_calls_;
        store_entries_ = entries;
        return StartTask(entries.size(), task_id, store_submit_status_);
    }

    kv::Status DeleteAsync(const std::vector<kv::CacheKey>& keys,
                           kv::TaskId& task_id, const kv::IoOptions&) override {
        ++delete_calls_;
        delete_keys_ = keys;
        return StartTask(keys.size(), task_id, delete_submit_status_);
    }

    bool Check(kv::TaskId) override { return true; }

    kv::Status Wait(kv::TaskId, std::uint64_t,
                    kv::TaskResult& result) override {
        ++wait_calls_;
        result.status = task_status_;
        // The adapter rejects a count mismatch, so mirror the submitted entry
        // count unless the test injected explicit per-entry statuses.
        result.entryStatus = entry_status_.empty()
                                 ? std::vector<kv::Status>(pending_entries_)
                                 : entry_status_;
        return wait_status_;
    }

    kv::Status RegisterRegions(
        const std::vector<kv::MemoryRegion>& regions,
        std::vector<kv::RegisteredMemory>& out) override {
        ++register_calls_;
        registered_regions_.insert(registered_regions_.end(), regions.begin(),
                                   regions.end());
        out.clear();
        if (!register_status_.ok()) return register_status_;
        for (const auto& region : regions) {
            out.push_back(kv::RegisteredMemory{region, next_handle_++, 1});
        }
        return {};
    }

    kv::Status UnregisterRegions(
        const std::vector<kv::MRHandle>& handles) override {
        ++unregister_calls_;
        unregistered_handles_.insert(unregistered_handles_.end(),
                                     handles.begin(), handles.end());
        return unregister_status_;
    }

    int TotalSdkCalls() const {
        return query_calls_ + load_calls_ + store_calls_ + delete_calls_ +
               wait_calls_ + register_calls_ + unregister_calls_;
    }

    // Injectable results.
    kv::Status task_status_{};
    kv::Status wait_status_{};
    kv::Status register_status_{};
    kv::Status unregister_status_{};
    kv::Status query_submit_status_{};
    kv::Status load_submit_status_{};
    kv::Status store_submit_status_{};
    kv::Status delete_submit_status_{};
    std::vector<kv::Status> entry_status_;

    // Recorded calls.
    int query_calls_ = 0;
    int load_calls_ = 0;
    int store_calls_ = 0;
    int delete_calls_ = 0;
    int wait_calls_ = 0;
    int register_calls_ = 0;
    int unregister_calls_ = 0;
    int shutdown_calls_ = 0;
    std::vector<kv::CacheKey> query_keys_;
    std::vector<kv::CacheKey> delete_keys_;
    std::vector<kv::KVBuffer> load_entries_;
    std::vector<kv::KVBuffer> store_entries_;
    std::vector<kv::MemoryRegion> registered_regions_;
    std::vector<kv::MRHandle> unregistered_handles_;

   private:
    kv::Status StartTask(size_t entry_count, kv::TaskId& task_id,
                         const kv::Status& submit_status) {
        pending_entries_ = entry_count;
        if (!submit_status.ok()) return submit_status;
        task_id = next_task_id_++;
        return {};
    }

    size_t pending_entries_ = 0;
    kv::MRHandle next_handle_ = 1;
    kv::TaskId next_task_id_ = 1;
};

// Owns the fake client and an adapter wired to it. The injected-client
// constructor is the adapter's test seam: Init() then only validates the
// adapter config and never calls the SDK's Init.
struct Harness {
    std::unique_ptr<KvClientObjectStorageAdapter> adapter;
    FakeKvClient* fake = nullptr;
    bool initialized = false;

    explicit Harness(bool init = true) {
        auto client = std::make_unique<FakeKvClient>();
        fake = client.get();
        adapter = std::make_unique<KvClientObjectStorageAdapter>(
            KvClientAdapterConfig{"/tmp/kvclient.conf"}, std::move(client));
        if (init) {
            auto result = adapter->Init();
            // An EXPECT in a constructor cannot abort the test, so publish the
            // outcome and let each test assert on it.
            initialized = result.has_value();
        }
    }
};

// Fails the current test unless the adapter reached the initialized state.
void RequireInitialized(const Harness& harness) {
    ASSERT_TRUE(harness.initialized) << "adapter Init() must succeed";
}

kv::BinaryKey ExpectedKey(std::string_view logical_key) {
    uint64_t hash = XXH64(logical_key.data(), logical_key.size(), kHashSeed);
    if (hash == 0) hash = kHashSeed;
    kv::BinaryKey key{};
    for (size_t i = 0; i < sizeof(hash); ++i) {
        key[i] = static_cast<std::byte>(hash >> (8 * i));
    }
    return key;
}

TEST(KvClientAdapterTest, MakeCacheKeyIsStableAndLittleEndian) {
    Harness h;
    RequireInitialized(h);
    const std::vector<std::string> keys{"alpha", "alpha"};
    auto deleted = h.adapter->DeleteBatch(keys);
    ASSERT_EQ(deleted.size(), 2u);
    ASSERT_EQ(h.fake->delete_keys_.size(), 2u);

    const uint64_t hash = XXH64("alpha", 5, kHashSeed);
    const auto expected = ExpectedKey("alpha");
    for (const auto& captured : h.fake->delete_keys_) {
        const auto* binary = captured.TryGetBinaryKey();
        ASSERT_NE(binary, nullptr);
        // The same logical key always encodes identically.
        EXPECT_EQ(*binary, expected);
        // Explicit little-endian packing: low byte first.
        EXPECT_EQ(static_cast<uint8_t>((*binary)[0]), hash & 0xFF);
        EXPECT_EQ(static_cast<uint8_t>((*binary)[7]), (hash >> 56) & 0xFF);
    }
}

TEST(KvClientAdapterTest, RegisterBuffersRegistersAndReleases) {
    Harness h;
    RequireInitialized(h);
    std::vector<char> first(64, 'a'), second(32, 'b');
    iovec first_iov{first.data(), first.size()};
    iovec second_iov{second.data(), second.size()};
    std::vector<ObjectPutRequest> requests{{"k1", &first_iov, 1},
                                           {"k2", &second_iov, 1}};

    auto results = h.adapter->PutBatch(requests);
    ASSERT_EQ(results.size(), 2u);
    EXPECT_TRUE(results[0]);
    EXPECT_TRUE(results[1]);

    // One registration per participating request, covering its whole range.
    EXPECT_EQ(h.fake->register_calls_, 2);
    ASSERT_EQ(h.fake->registered_regions_.size(), 2u);
    EXPECT_EQ(h.fake->registered_regions_[0].addr,
              reinterpret_cast<uint64_t>(first.data()));
    EXPECT_EQ(h.fake->registered_regions_[0].size, 64u);
    EXPECT_EQ(h.fake->registered_regions_[1].addr,
              reinterpret_cast<uint64_t>(second.data()));
    EXPECT_EQ(h.fake->registered_regions_[1].size, 32u);

    // Every handle is released once, in a single batched call.
    EXPECT_EQ(h.fake->unregister_calls_, 1);
    ASSERT_EQ(h.fake->unregistered_handles_.size(), 2u);
    EXPECT_EQ(h.fake->unregistered_handles_[0], 1u);
    EXPECT_EQ(h.fake->unregistered_handles_[1], 2u);

    // Handles reach the SDK entries untouched; mrKey stays for the SDK to fill.
    ASSERT_EQ(h.fake->store_entries_.size(), 2u);
    EXPECT_EQ(h.fake->store_entries_[0].buffer.handle, 1u);
    EXPECT_EQ(h.fake->store_entries_[1].buffer.handle, 2u);
    EXPECT_FALSE(h.fake->store_entries_[0].mrKey.has_value());
}

TEST(KvClientAdapterTest, RejectsEmptyKeyZeroLengthAndDuplicate) {
    Harness h;
    RequireInitialized(h);
    std::vector<char> payload(16, 'x');
    iovec good{payload.data(), payload.size()};
    iovec zero_length{payload.data(), 0};
    iovec null_segment{nullptr, 8};

    std::vector<ObjectPutRequest> requests{
        {"", &good, 1},               // empty key
        {"dup", &good, 1},            // first occurrence
        {"dup", &good, 1},            // duplicate of the previous entry
        {"zero", &zero_length, 1},    // zero-length segment
        {"null", &null_segment, 1},   // null segment
        {"no-segments", nullptr, 0},  // no iovec at all
    };
    auto results = h.adapter->PutBatch(requests);
    ASSERT_EQ(results.size(), 6u);
    EXPECT_EQ(results[0].error(), ErrorCode::INVALID_PARAMS);
    EXPECT_TRUE(results[1]) << "the first occurrence must be accepted";
    EXPECT_EQ(results[2].error(), ErrorCode::INVALID_PARAMS);
    for (size_t i = 3; i < 6; ++i) {
        ASSERT_FALSE(results[i]) << "index " << i;
        EXPECT_EQ(results[i].error(), ErrorCode::INVALID_PARAMS);
    }
    // Only the accepted entry was registered and submitted.
    EXPECT_EQ(h.fake->register_calls_, 1);
    ASSERT_EQ(h.fake->store_entries_.size(), 1u);
}

TEST(KvClientAdapterTest, RejectsNonContiguousSegments) {
    Harness h;
    RequireInitialized(h);
    // Two independent allocations cannot form one registered range.
    std::vector<char> first(16, 'a'), second(16, 'b');
    std::vector<iovec> iovs{{first.data(), first.size()},
                            {second.data(), second.size()}};
    std::vector<ObjectPutRequest> requests{{"split", iovs.data(), 2}};

    auto results = h.adapter->PutBatch(requests);
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].error(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(h.fake->register_calls_, 0);
    EXPECT_EQ(h.fake->store_calls_, 0);
}

TEST(KvClientAdapterTest, AcceptsContiguousSegments) {
    Harness h;
    RequireInitialized(h);
    std::vector<char> payload(32, 'c');
    // Two iovecs covering one contiguous allocation.
    std::vector<iovec> iovs{{payload.data(), 12}, {payload.data() + 12, 20}};
    std::vector<ObjectPutRequest> requests{{"whole", iovs.data(), 2}};

    auto results = h.adapter->PutBatch(requests);
    ASSERT_EQ(results.size(), 1u);
    EXPECT_TRUE(results[0]);
    ASSERT_EQ(h.fake->store_entries_.size(), 1u);
    EXPECT_EQ(h.fake->store_entries_[0].buffer.region.addr,
              reinterpret_cast<uint64_t>(payload.data()));
    EXPECT_EQ(h.fake->store_entries_[0].buffer.region.size, 32u);
}

TEST(KvClientAdapterTest, RegisterFailureIsolatesEntry) {
    Harness h;
    RequireInitialized(h);
    h.fake->register_status_ = Injected(kv::StatusCode::PARTIAL_FAILED);
    std::vector<char> payload(16, 'x');
    iovec iov{payload.data(), payload.size()};
    std::vector<ObjectPutRequest> requests{{"k1", &iov, 1}, {"k2", &iov, 1}};

    auto results = h.adapter->PutBatch(requests);
    ASSERT_EQ(results.size(), 2u);
    for (const auto& result : results) {
        ASSERT_FALSE(result);
        EXPECT_EQ(result.error(), ErrorCode::DFS_SERVICE_UNAVAILABLE);
    }
    // Nothing was submitted, and nothing was registered to release.
    EXPECT_EQ(h.fake->store_calls_, 0);
    EXPECT_EQ(h.fake->unregister_calls_, 0);
}

TEST(KvClientAdapterTest, RegisterFailureDoesNotClaimKey) {
    Harness h;
    RequireInitialized(h);
    std::vector<char> payload(16, 'x');
    iovec iov{payload.data(), payload.size()};

    // A rejected registration must not occupy the duplicate-key slot, so the
    // same key can still be written once registration succeeds.
    h.fake->register_status_ = Injected(kv::StatusCode::PARTIAL_FAILED);
    std::vector<ObjectPutRequest> first{{"same", &iov, 1}};
    EXPECT_FALSE(h.adapter->PutBatch(first)[0]);

    h.fake->register_status_ = {};
    std::vector<ObjectPutRequest> second{{"same", &iov, 1}};
    EXPECT_TRUE(h.adapter->PutBatch(second)[0]);
}

TEST(KvClientAdapterTest, SingleEntryUsesBatchPath) {
    Harness h;
    RequireInitialized(h);
    std::string value = "payload";

    EXPECT_TRUE(h.adapter->Put("single", value));
    EXPECT_EQ(h.fake->store_calls_, 1);
    ASSERT_EQ(h.fake->store_entries_.size(), 1u);
    EXPECT_EQ(h.fake->store_entries_[0].buffer.region.size, value.size());
    EXPECT_EQ(h.fake->register_calls_, 1);
    EXPECT_EQ(h.fake->unregister_calls_, 1);

    iovec iov{value.data(), value.size()};
    EXPECT_TRUE(h.adapter->PutV("vectored", &iov, 1));
    EXPECT_EQ(h.fake->store_calls_, 2);

    // An empty span has zero length and is rejected before any SDK call.
    const int calls_before = h.fake->TotalSdkCalls();
    auto empty = h.adapter->Put("empty", std::span<const char>{});
    ASSERT_FALSE(empty);
    EXPECT_EQ(empty.error(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(h.fake->TotalSdkCalls(), calls_before);
}

TEST(KvClientAdapterTest, WriteTaskErrorMapsAll) {
    Harness h;
    RequireInitialized(h);
    h.fake->task_status_ = Injected(kv::StatusCode::CONNECTION_ERROR);
    std::vector<char> first(16, 'a'), second(16, 'b');
    iovec first_iov{first.data(), first.size()};
    iovec second_iov{second.data(), second.size()};
    std::vector<ObjectPutRequest> requests{{"k1", &first_iov, 1},
                                           {"k2", &second_iov, 1}};

    auto results = h.adapter->PutBatch(requests);
    ASSERT_EQ(results.size(), 2u);
    for (const auto& result : results) {
        ASSERT_FALSE(result);
        EXPECT_EQ(result.error(), ErrorCode::DFS_SERVICE_UNAVAILABLE);
    }
    // Handles are still released after a failed task.
    EXPECT_EQ(h.fake->unregister_calls_, 1);
    EXPECT_EQ(h.fake->unregistered_handles_.size(), 2u);
}

TEST(KvClientAdapterTest, WriteSubmitFailureKeepsGuardErrors) {
    Harness h;
    RequireInitialized(h);
    h.fake->store_submit_status_ = Injected(kv::StatusCode::RESOURCE_BUSY);
    std::vector<char> payload(16, 'x');
    iovec iov{payload.data(), payload.size()};
    std::vector<ObjectPutRequest> requests{{"k1", &iov, 1}, {"", &iov, 1}};

    auto results = h.adapter->PutBatch(requests);
    ASSERT_EQ(results.size(), 2u);
    EXPECT_EQ(results[0].error(), ErrorCode::NO_AVAILABLE_HANDLE);
    // The guard error set before submission survives the submit-failure
    // mapping, which only covers participating entries.
    EXPECT_EQ(results[1].error(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(h.fake->wait_calls_, 0);
    EXPECT_EQ(h.fake->unregistered_handles_.size(), 1u);
}

TEST(KvClientAdapterTest, WriteEntryFailureKeepsOtherEntriesAndReleases) {
    Harness h;
    RequireInitialized(h);
    h.fake->task_status_ = Injected(kv::StatusCode::PARTIAL_FAILED);
    h.fake->entry_status_ = {kv::Status::OK(),
                             Injected(kv::StatusCode::ENTRY_RETRY_ADVISED)};
    std::vector<char> first(16, 'a'), second(16, 'b');
    iovec first_iov{first.data(), first.size()};
    iovec second_iov{second.data(), second.size()};
    std::vector<ObjectPutRequest> requests{{"k1", &first_iov, 1},
                                           {"k2", &second_iov, 1}};

    auto results = h.adapter->PutBatch(requests);
    ASSERT_EQ(results.size(), 2u);
    EXPECT_TRUE(results[0]);
    // A retryable storage condition is a service problem, not an internal bug.
    EXPECT_EQ(results[1].error(), ErrorCode::DFS_SERVICE_UNAVAILABLE);
    EXPECT_EQ(h.fake->unregistered_handles_.size(), 2u);
}

TEST(KvClientAdapterTest, ReadSucceeds) {
    Harness h;
    RequireInitialized(h);
    std::vector<char> buffer(24, '\0');
    std::vector<ObjectGetRequest> requests{
        {"k1", buffer.data(), buffer.size()}};

    auto results = h.adapter->GetBatch(requests);
    ASSERT_EQ(results.size(), 1u);
    ASSERT_TRUE(results[0]);
    // Success reports the requested length, not an SDK transfer count.
    EXPECT_EQ(*results[0], buffer.size());
    EXPECT_EQ(h.fake->load_calls_, 1);
    EXPECT_EQ(h.fake->store_calls_, 0);
    EXPECT_EQ(h.fake->wait_calls_, 1);
    ASSERT_EQ(h.fake->load_entries_.size(), 1u);
    EXPECT_EQ(h.fake->load_entries_[0].buffer.region.size, buffer.size());
    EXPECT_EQ(h.fake->load_entries_[0].buffer.handle, 1u);
    // Registered once and released before returning.
    EXPECT_EQ(h.fake->register_calls_, 1);
    EXPECT_EQ(h.fake->unregistered_handles_.size(), 1u);
}

TEST(KvClientAdapterTest, ReadSubmitFailureLeavesBufferUntouched) {
    Harness h;
    RequireInitialized(h);
    h.fake->load_submit_status_ = Injected(kv::StatusCode::CONNECTION_ERROR);
    std::vector<char> buffer(16, '\0');
    std::vector<ObjectGetRequest> requests{
        {"k1", buffer.data(), buffer.size()}};

    auto results = h.adapter->GetBatch(requests);
    ASSERT_EQ(results.size(), 1u);
    ASSERT_FALSE(results[0]);
    EXPECT_EQ(results[0].error(), ErrorCode::DFS_SERVICE_UNAVAILABLE);
    EXPECT_EQ(h.fake->wait_calls_, 0);
    EXPECT_EQ(buffer, std::vector<char>(16, '\0'));
    EXPECT_EQ(h.fake->unregistered_handles_.size(), 1u);
}

TEST(KvClientAdapterTest, ReadTaskErrorMapsAll) {
    Harness h;
    RequireInitialized(h);
    h.fake->task_status_ = Injected(kv::StatusCode::CQE_IO_TIMEOUT);
    std::vector<char> first(16, '\0'), second(16, '\0');
    std::vector<ObjectGetRequest> requests{
        {"k1", first.data(), first.size()},
        {"k2", second.data(), second.size()}};

    auto results = h.adapter->GetBatch(requests);
    ASSERT_EQ(results.size(), 2u);
    for (const auto& result : results) {
        ASSERT_FALSE(result);
        EXPECT_EQ(result.error(), ErrorCode::RPC_TIMEOUT);
    }
    EXPECT_EQ(h.fake->unregistered_handles_.size(), 2u);
}

TEST(KvClientAdapterTest, ReadMissIsObjectNotFound) {
    Harness h;
    RequireInitialized(h);
    // A LOAD miss arrives as a per-entry code under a PARTIAL_FAILED task,
    // which is still terminal for per-entry purposes.
    h.fake->task_status_ = Injected(kv::StatusCode::PARTIAL_FAILED);
    h.fake->entry_status_ = {Injected(kv::StatusCode::ENTRY_KEY_NOT_FOUND)};
    std::vector<char> buffer(16, '\0');
    std::vector<ObjectGetRequest> requests{
        {"missing", buffer.data(), buffer.size()}};

    auto results = h.adapter->GetBatch(requests);
    ASSERT_EQ(results.size(), 1u);
    ASSERT_FALSE(results[0]);
    EXPECT_EQ(results[0].error(), ErrorCode::OBJECT_NOT_FOUND);
}

TEST(KvClientAdapterTest, ReadDataNotExistIsObjectNotFound) {
    Harness h;
    RequireInitialized(h);
    h.fake->task_status_ = Injected(kv::StatusCode::PARTIAL_FAILED);
    h.fake->entry_status_ = {Injected(kv::StatusCode::ENTRY_DATA_NOT_EXIST)};
    std::vector<char> buffer(16, '\0');
    std::vector<ObjectGetRequest> requests{
        {"missing", buffer.data(), buffer.size()}};

    auto results = h.adapter->GetBatch(requests);
    ASSERT_EQ(results.size(), 1u);
    ASSERT_FALSE(results[0]);
    EXPECT_EQ(results[0].error(), ErrorCode::OBJECT_NOT_FOUND);
}

TEST(KvClientAdapterTest, ExistsReturnsEntryCode) {
    Harness h;
    RequireInitialized(h);
    // A hit reports ENTRY_KEY_EXIST, which is an Error status, so ok() cannot
    // be used to detect it.
    h.fake->task_status_ = {};
    h.fake->entry_status_ = {Injected(kv::StatusCode::ENTRY_KEY_EXIST)};
    auto hit = h.adapter->Exists("present");
    ASSERT_TRUE(hit.has_value()) << static_cast<int>(hit.error());
    EXPECT_TRUE(*hit);
    EXPECT_EQ(h.fake->query_calls_, 1);
    // Existence queries never register a buffer.
    EXPECT_EQ(h.fake->register_calls_, 0);

    // A miss leaves the task PARTIAL_FAILED yet still carries a trustworthy
    // per-entry result, so it must not be treated as a failure.
    h.fake->task_status_ = Injected(kv::StatusCode::PARTIAL_FAILED);
    h.fake->entry_status_ = {Injected(kv::StatusCode::ENTRY_KEY_NOT_EXIST)};
    auto miss = h.adapter->Exists("absent");
    ASSERT_TRUE(miss.has_value()) << static_cast<int>(miss.error());
    EXPECT_FALSE(*miss);
}

TEST(KvClientAdapterTest, ExistsErrors) {
    Harness h;
    RequireInitialized(h);
    const int before = h.fake->TotalSdkCalls();
    auto empty = h.adapter->Exists("");
    ASSERT_FALSE(empty);
    EXPECT_EQ(empty.error(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(h.fake->TotalSdkCalls(), before) << "empty key must not query";

    h.fake->query_submit_status_ = Injected(kv::StatusCode::CONNECTION_ERROR);
    auto submit_failed = h.adapter->Exists("key");
    ASSERT_FALSE(submit_failed);
    EXPECT_EQ(submit_failed.error(), ErrorCode::DFS_SERVICE_UNAVAILABLE);

    h.fake->query_submit_status_ = {};
    h.fake->task_status_ = Injected(kv::StatusCode::PARTIAL_FAILED);
    h.fake->entry_status_ = {Injected(kv::StatusCode::IO_ERROR)};
    auto entry_failed = h.adapter->Exists("key");
    ASSERT_FALSE(entry_failed);
    EXPECT_EQ(entry_failed.error(), ErrorCode::DFS_SERVICE_UNAVAILABLE);
}

TEST(KvClientAdapterTest, SingleEntryReadUsesBatchPath) {
    Harness h;
    std::vector<char> buffer(16, '\0');

    auto got = h.adapter->Get("single", buffer.data(), buffer.size());
    ASSERT_TRUE(got.has_value()) << static_cast<int>(got.error());
    EXPECT_EQ(*got, buffer.size());
    // Get is a single-element GetBatch: one load, one read-only registration.
    EXPECT_EQ(h.fake->load_calls_, 1);
    EXPECT_EQ(h.fake->store_calls_, 0);
    ASSERT_EQ(h.fake->load_entries_.size(), 1u);
    EXPECT_EQ(h.fake->load_entries_[0].buffer.region.size, buffer.size());
    EXPECT_EQ(h.fake->register_calls_, 1);
    EXPECT_EQ(h.fake->unregistered_handles_.size(), 1u);

    // A null destination has zero usable length and is rejected by the guard.
    const int calls_before = h.fake->TotalSdkCalls();
    auto null_buf = h.adapter->Get("single", nullptr, 16);
    ASSERT_FALSE(null_buf);
    EXPECT_EQ(null_buf.error(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(h.fake->TotalSdkCalls(), calls_before);
}

TEST(KvClientAdapterTest, OverwriteDoesNotPreDelete) {
    Harness h;
    RequireInitialized(h);
    std::vector<char> payload(16, 'x');
    iovec iov{payload.data(), payload.size()};
    std::vector<ObjectPutRequest> requests{{"same-key", &iov, 1}};

    EXPECT_TRUE(h.adapter->PutBatch(requests)[0]);
    EXPECT_TRUE(h.adapter->PutBatch(requests)[0]);
    EXPECT_EQ(h.fake->store_calls_, 2);
    EXPECT_EQ(h.fake->delete_calls_, 0) << "StoreAsync is an upsert";
}

TEST(KvClientAdapterTest, UninitializedBatchFailsSafely) {
    Harness h(/*init=*/false);
    std::vector<char> payload(16, 'x');
    iovec iov{payload.data(), payload.size()};
    std::vector<ObjectPutRequest> puts{{"k1", &iov, 1}};
    std::vector<ObjectGetRequest> gets{{"k1", payload.data(), payload.size()}};

    auto put_results = h.adapter->PutBatch(puts);
    ASSERT_EQ(put_results.size(), 1u);
    ASSERT_FALSE(put_results[0]);
    EXPECT_EQ(put_results[0].error(), ErrorCode::DFS_SERVICE_UNAVAILABLE);

    auto get_results = h.adapter->GetBatch(gets);
    ASSERT_EQ(get_results.size(), 1u);
    ASSERT_FALSE(get_results[0]);
    EXPECT_EQ(get_results[0].error(), ErrorCode::DFS_SERVICE_UNAVAILABLE);

    auto exists = h.adapter->Exists("k1");
    ASSERT_FALSE(exists);
    EXPECT_EQ(exists.error(), ErrorCode::DFS_SERVICE_UNAVAILABLE);

    EXPECT_EQ(h.fake->TotalSdkCalls(), 0) << "no SDK call before Init";
}

TEST(KvClientAdapterTest, UnregisterFailureIsLoggedNotFatal) {
    Harness h;
    RequireInitialized(h);
    h.fake->unregister_status_ = Injected(kv::StatusCode::CONNECTION_ERROR);
    std::vector<char> payload(16, 'x');
    iovec iov{payload.data(), payload.size()};
    std::vector<ObjectPutRequest> requests{{"k1", &iov, 1}};

    auto results = h.adapter->PutBatch(requests);
    ASSERT_EQ(results.size(), 1u);
    // The write succeeded; a failed release neither changes the result nor is
    // retried.
    EXPECT_TRUE(results[0]);
    EXPECT_EQ(h.fake->unregister_calls_, 1);
}

TEST(KvClientAdapterTest, ListKeysIsUnsupported) {
    Harness h;
    RequireInitialized(h);
    auto keys = h.adapter->ListKeys();
    ASSERT_FALSE(keys);
    EXPECT_EQ(keys.error(), ErrorCode::NOT_SUPPORTED);
    EXPECT_STREQ(h.adapter->GetName(), "kvclient");
}

}  // namespace
}  // namespace mooncake
