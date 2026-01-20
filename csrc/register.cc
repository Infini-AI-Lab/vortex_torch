#include "register.h"



PYBIND11_MODULE(vortex_torch_C, m){
        m.def("sglang_plan_decode",             &sglang_plan_decode);
        m.def("sglang_plan_prefill",            &sglang_plan_prefill);
        m.def("Chunkwise_NH2HN_Transpose",      &Chunkwise_NH2HN_Transpose);
        m.def("Chunkwise_HN2NH_Transpose",      &Chunkwise_HN2NH_Transpose);
        m.def("topk_output",                    &topk_output);
        m.def("sglang_plan_decode_fa3",         &sglang_plan_decode_fa3);
        m.def("sglang_plan_prefill_fa3",        &sglang_plan_prefill_fa3);
        m.def("Chunkwise_HN2NH_Transpose_FA3",  &Chunkwise_HN2NH_Transpose_FA3);
        m.def("unified_reduce",                 &unified_reduce);
        m.def("store_kv_unified",               &store_kv_unified);
        m.def("allocate_pages_lru_warp_with_indptr", &allocate_pages_lru_warp_with_indptr);
        m.def("allocate_pages_hybrid", &allocate_pages_hybrid);
        m.def("init_hybrid_structures",         &init_hybrid_structures);
        m.def("copy_kv",                        &copy_kv);
}
