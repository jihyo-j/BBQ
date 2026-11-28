#pragma once
// Stand-in for exp/baselines/qdrant_tq_ffi/qdrant_tq_ffi.h, used when bbq_dynamic_insert is built without the
// TurboQuant baseline (BBQ_WITH_TQ=0, the default of `bbq/build.sh dynamic`).
//
// The BBQ insertion paths (--online-graph-add fp | bq | adcbq) never call into the codec. Only the TurboQuant-1bit
// comparison arm of Fig 18 (--online-graph-add tq1) does; bbq_dynamic_insert rejects that mode up front in this build.
// The free functions are no-ops because destructors call them with null handles; every other entry point fails
// with an explanatory error.

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

typedef struct QdrantTqIndex QdrantTqIndex;
typedef struct QdrantTqQuery QdrantTqQuery;

#define BBQ_TQ_STUB_ERROR \
    "built without the TurboQuant baseline; rebuild with BBQ_WITH_TQ=1 ./bbq/build.sh dynamic"

static inline const char* qdrant_tq_last_error(void) { return BBQ_TQ_STUB_ERROR; }
static inline QdrantTqIndex* qdrant_tq_load(const char*, const char*) { return nullptr; }
static inline bool qdrant_tq_append(QdrantTqIndex*, const float*, size_t, size_t, size_t) { return false; }
static inline void qdrant_tq_free(QdrantTqIndex*) {}
static inline QdrantTqQuery* qdrant_tq_query_new(const QdrantTqIndex*, const float*, size_t) { return nullptr; }
static inline void qdrant_tq_query_free(QdrantTqQuery*) {}
static inline bool qdrant_tq_query_export_bits1_wide_planes(
        const QdrantTqQuery*, int8_t*, size_t, float*, int64_t*, float*, float*) {
    return false;
}
static inline size_t qdrant_tq_code_size(const QdrantTqIndex*) { return 0; }
static inline const uint8_t* qdrant_tq_codes(const QdrantTqIndex*, size_t* byte_len) {
    if (byte_len) *byte_len = 0;
    return nullptr;
}
