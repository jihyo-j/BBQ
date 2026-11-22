#pragma once

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct QdrantTqIndex QdrantTqIndex;
typedef struct QdrantTqQuery QdrantTqQuery;

const char* qdrant_tq_last_error(void);
QdrantTqIndex* qdrant_tq_build(
        const float* data,
        size_t rows,
        size_t dim,
        int32_t distance_code,
        size_t num_threads,
        const char* data_path,
        const char* meta_path);
QdrantTqIndex* qdrant_tq_load(const char* data_path, const char* meta_path);
bool qdrant_tq_append(
        QdrantTqIndex* index,
        const float* data,
        size_t rows,
        size_t dim,
        size_t num_threads);
void qdrant_tq_free(QdrantTqIndex* index);
QdrantTqQuery* qdrant_tq_query_new(
        const QdrantTqIndex* index, const float* query, size_t dim);
void qdrant_tq_query_free(QdrantTqQuery* query);
bool qdrant_tq_query_export_bits1_wide(
        const QdrantTqQuery* query,
        int16_t* signed_query,
        size_t signed_query_len,
        float* postprocess_scale,
        int64_t* bias_correction,
        float* ec_correction,
        float* l2_norm);
bool qdrant_tq_query_export_bits1_wide_planes(
        const QdrantTqQuery* query,
        int8_t* query_planes,
        size_t query_planes_len,
        float* postprocess_scale,
        int64_t* bias_correction,
        float* ec_correction,
        float* l2_norm);
float qdrant_tq_score(
        const QdrantTqIndex* index, const QdrantTqQuery* query, uint32_t id);
float qdrant_tq_score_internal(
        const QdrantTqIndex* index, uint32_t left, uint32_t right);
size_t qdrant_tq_code_size(const QdrantTqIndex* index);
const uint8_t* qdrant_tq_codes(const QdrantTqIndex* index, size_t* byte_len);

#ifdef __cplusplus
}
#endif

