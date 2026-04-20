#include "register.h"



PYBIND11_MODULE(vortex_torch_C, m){
        m.def("sglang_plan_decode",             &sglang_plan_decode);
        m.def("sglang_plan_prefill",            &sglang_plan_prefill);
        m.def("Chunkwise_NH2HN_Transpose",      &Chunkwise_NH2HN_Transpose);
        m.def("Chunkwise_HN2NH_Transpose",      &Chunkwise_HN2NH_Transpose);
        m.def("topk_output",                    &topk_output);
        m.def("topk_output_sglang",             &topk_output_sglang,
              py::arg("x"), py::arg("dense_kv_indptr"), py::arg("sparse_kv_indptr"),
              py::arg("dense_kv_indices"), py::arg("sparse_kv_indices"),
              py::arg("eff_batch_size"), py::arg("topk_val"),
              py::arg("reserved_bos"), py::arg("reserved_eos"),
              py::arg("max_num_pages"));
        m.def("topk_output_sglang_ori",         &topk_output_sglang_ori,
              py::arg("x"), py::arg("dense_kv_indptr"),
              py::arg("indices_out"),
              py::arg("eff_batch_size"), py::arg("topk_val"),
              py::arg("reserved_bos"), py::arg("reserved_eos"),
              py::arg("max_num_pages"));
        m.def("topk_output_sglang_fused",       &topk_output_sglang_fused,
              py::arg("x"), py::arg("dense_kv_indptr"), py::arg("sparse_kv_indptr"),
              py::arg("dense_kv_indices"), py::arg("sparse_kv_indices"),
              py::arg("eff_batch_size"), py::arg("topk_val"),
              py::arg("reserved_bos"), py::arg("reserved_eos"),
              py::arg("max_num_pages"),
              py::arg("mapping_mode"),
              py::arg("mapping_power"),
              py::arg("mapping_lut") = py::none(),
              py::arg("mapping_quantiles") = py::none());
        m.def("fast_fused_topk_merge",          &fast_fused_topk_merge,
              py::arg("score"), py::arg("global_topk_indices"),
              py::arg("batch_size"), py::arg("num_chunks"),
              py::arg("chunk_size"), py::arg("topk_val"),
              py::arg("mapping_mode"), py::arg("mapping_power"));
        m.def("fast_cluster_topk_merge",        &fast_cluster_topk_merge,
              py::arg("score"), py::arg("global_topk_indices"),
              py::arg("batch_size"), py::arg("num_chunks"),
              py::arg("chunk_size"), py::arg("topk_val"),
              py::arg("mapping_mode"), py::arg("mapping_power"));
        m.def("topk_remap_only",                &topk_remap_only,
              py::arg("x"), py::arg("dense_kv_indptr"),
              py::arg("remapped"),
              py::arg("eff_batch_size"),
              py::arg("reserved_bos"), py::arg("reserved_eos"),
              py::arg("mapping_mode"),
              py::arg("mapping_power"));
        m.def("topk_profile_histogram",        &topk_profile_histogram,
              py::arg("x"), py::arg("dense_kv_indptr"),
              py::arg("histograms"), py::arg("eff_batch_size"),
              py::arg("reserved_bos"), py::arg("reserved_eos"),
              py::arg("mapping_mode") = 0,
              py::arg("mapping_power") = 0.5,
              py::arg("mapping_lut") = py::none(),
              py::arg("mapping_quantiles") = py::none());
        m.def("topk_profile_counters",         &topk_profile_counters,
              py::arg("x"), py::arg("dense_kv_indptr"), py::arg("sparse_kv_indptr"),
              py::arg("dense_kv_indices"), py::arg("sparse_kv_indices"),
              py::arg("counters"),
              py::arg("eff_batch_size"), py::arg("topk_val"),
              py::arg("reserved_bos"), py::arg("reserved_eos"),
              py::arg("max_num_pages"),
              py::arg("mapping_mode") = 0,
              py::arg("mapping_power") = 0.5,
              py::arg("mapping_lut") = py::none(),
              py::arg("mapping_quantiles") = py::none());
        m.def("sglang_plan_decode_fa3",         &sglang_plan_decode_fa3);
        m.def("sglang_plan_prefill_fa3",        &sglang_plan_prefill_fa3);
        m.def("Chunkwise_HN2NH_Transpose_FA3",  &Chunkwise_HN2NH_Transpose_FA3);
}
