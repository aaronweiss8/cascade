#include <cascade/config.h>
#include <cascade/data_path_logic_interface.hpp>
#include <iostream>
#include <mxnet-cpp/MxNetCpp.h>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include "cnn_classifier_dpl.hpp"

/**
 * This is an example for the filter/trigger data path logic with ML model serving. In this example, we process incoming
 * photos/video frames based on its keys. If a key matches "pet/...", this object will trigger a categorizer to tell
 * the breed of the pet; If a key matches "flower/...", this object will trigger a categorizer to tell the name of the
 * flower picture. The result will be stored in a persisted cascade subgroup.
 *
 * We create an environment with the following layout:
 * - Subgroup VolatileCascadeStoreWithStringKey:0 - categorizer subgroup processes the incoming data and send the result to PersistentCascadeStoreWithStringKey:0. VolatileCascadeStoreWithStringKey:0 has one
 *                     two-nodes shard for this subgroup. The two nodes process partition the key space bashed on
 *                     a hash function.
 * - Subgroup PersistentCascadeStoreWithStringKey:0 - persisted tag shard, which stores all the tags. The keys mirror those in VolatileCascadeStoreWithStringKey:0. PersistentCascadeStoreWithStringKey:0 has one
 *                     three-node shard for this subgroup.
 *
 * TODO: critical data path --> filter/dispatcher; off critical data path --> trigger
 */
namespace derecho{
namespace cascade{

#define PET_PREFIX   "/pet-model"
#define FLOWER_PREFIX "flower-model"
#define MY_UUID     "48e60f7c-8500-11eb-8755-0242ac110003"
#define MY_DESC     "Demo classifier for pets and flowers."

void initialize(ICascadeContext* ctxt) {
    std::cout << "[cnn_classifier example]: initialize the data path library here." << std::endl;
}

void release(ICascadeContext* ctxt) {
    std::cout << "[cnn_classifier example]: destroy data path environment before exit." << std::endl;
}

std::unordered_set<std::string> list_prefixes() {
    return {PET_PREFIX, FLOWER_PREFIX};
}

std::string get_uuid() {
    return MY_UUID;
}

std::string get_description() {
    return MY_DESC;
}

#define AT_UNKNOWN      (0)
#define AT_PET_BREED    (1)
#define AT_FLOWER_NAME  (2)
/**
 * StaticActionTable translates the key to action type:
 *
 */
class StaticActionTable {
    struct _static_action_table_entry {
        const std::string prefix;
        const uint64_t action_id;
    };
    /* mapping from prefix to ACTION TYPE */
    const std::vector<struct _static_action_table_entry> table;
public:
    StaticActionTable() : table({{"pet",AT_PET_BREED},{"flower",AT_FLOWER_NAME}}) {}
    uint64_t to_action(const std::string& key) {
        uint64_t aid = AT_UNKNOWN;
        for (const auto& entry: table) {
            if (key.find(entry.prefix) == 0) {
                aid = entry.action_id;
                break;
            }
        }
        return aid;
    }
};
static StaticActionTable static_action_table;
/*
 * The image frame data in predefined 224x224 pixel format.
 */
class ImageFrame: public Blob {
public:
    std::string key;
    ImageFrame(const std::string& k, const Blob& other): Blob(other), key(k) {}
};

enum TypeFlag {
    kFloat32 = 0,
    kFloat64 = 1,
    kFloat16 = 2,
    kUint8 = 3,
    kInt32 = 4,
    kInt8 = 5,
    kInt64 = 6,
};

template <typename CascadeType>
class ClassifierFilter: public CriticalDataPathObserver<CascadeType> {
    virtual void operator() (const uint32_t sgidx,
                             const uint32_t shidx,
                             const typename CascadeType::KeyType& key,
                             const typename CascadeType::ObjectType& value, ICascadeContext* cascade_ctxt) {
        auto* ctxt = dynamic_cast<CascadeContext<VolatileCascadeStoreWithStringKey,PersistentCascadeStoreWithStringKey>*>(cascade_ctxt);
        if constexpr (std::is_convertible<typename CascadeType::KeyType,std::string>::value) {
                auto* ctxt = dynamic_cast<CascadeContext<VolatileCascadeStoreWithStringKey,PersistentCascadeStoreWithStringKey>*>(cascade_ctxt);
                size_t pos = key.rfind('/');
                std::string prefix;
                if (pos != std::string::npos) {
                    prefix = key.substr(0,pos);
                }
                auto handlers = ctxt->get_prefix_handlers(prefix);
                auto value_ptr = std::make_unique<ImageFrame>(value.get_key_ref(),value.blob);
                for(auto& handler : handlers) {
                    Action action(key,value.get_version(),handler.second,value_ptr);
                    ctxt->post(std::move(action));
                }
            }


    }
};

#define DPL_CONF_FLOWER_SYNSET  "CASCADE/flower_synset"
#define DPL_CONF_FLOWER_SYMBOL  "CASCADE/flower_symbol"
#define DPL_CONF_FLOWER_PARAMS  "CASCADE/flower_params"
#define DPL_CONF_PET_SYNSET  "CASCADE/pet_synset"
#define DPL_CONF_PET_SYMBOL  "CASCADE/pet_symbol"
#define DPL_CONF_PET_PARAMS  "CASCADE/pet_params"
#define DPL_CONF_USE_GPU        "CASCADE/use_gpu"

class InferenceEngine {
private:
    /**
     * the synset explains inference result.
     */
    std::vector<std::string> synset_vector;
    /**
     * symbol
     */
    mxnet::cpp::Symbol net;
    /**
     * argument parameters
     */
    std::map<std::string, mxnet::cpp::NDArray> args_map;
    /**
     * auxliary parameters
     */
    std::map<std::string, mxnet::cpp::NDArray> aux_map;
    /**
     * global ctx
     */
    const mxnet::cpp::Context& global_ctx;
    /**
     * the input shape
     */
    mxnet::cpp::Shape input_shape;
    /**
     * argument arrays
     */
    std::vector<mxnet::cpp::NDArray> arg_arrays;
    /**
     * gradient arrays
     */
    std::vector<mxnet::cpp::NDArray> grad_arrays;
    /**
     * ??
     */
    std::vector<mxnet::cpp::OpReqType> grad_reqs;
    /**
     * auxliary array
     */
    std::vector<mxnet::cpp::NDArray> aux_arrays;
    /**
     * client data
     */
    mxnet::cpp::NDArray client_data;
    /**
     * the work horse: mxnet executor
     */
    std::unique_ptr<mxnet::cpp::Executor> executor_pointer;    
    
    void load_synset(const std::string& synset_file) {
        dbg_default_trace("synset file="+synset_file);
        std::ifstream fin(synset_file);
        synset_vector.clear();
        for(std::string syn;std::getline(fin,syn);) {
            synset_vector.push_back(syn);
        }
        fin.close();
    }

    void load_symbol(const std::string& symbol_file) {
        dbg_default_trace("symbol file="+symbol_file);
        this->net = mxnet::cpp::Symbol::Load(symbol_file);
    }

    void load_params(const std::string& params_file) {
        dbg_default_trace("params file="+params_file);
        auto parameters = mxnet::cpp::NDArray::LoadToMap(params_file);
        for (const auto& kv : parameters) {
            if (kv.first.substr(0, 4) == "aux:") {
                auto name = kv.first.substr(4, kv.first.size() - 4);
                this->aux_map[name] = kv.second.Copy(global_ctx);
            } else if (kv.first.substr(0, 4) == "arg:") {
                auto name = kv.first.substr(4, kv.first.size() - 4);
                this->args_map[name] = kv.second.Copy(global_ctx);
            }
        }
        mxnet::cpp::NDArray::WaitAll();
        this->args_map["data"] = mxnet::cpp::NDArray(input_shape, global_ctx, false, kFloat32);
        mxnet::cpp::Shape label_shape(input_shape[0]);
        this->args_map["softmax_label"] = mxnet::cpp::NDArray(label_shape, global_ctx, false);
        this->client_data = mxnet::cpp::NDArray(input_shape, global_ctx, false, kFloat32);
    }

    bool load_model(const std::string& synset_file,
                    const std::string& symbol_file,
                    const std::string& params_file) {
		try {
            load_synset(synset_file);
            load_symbol(symbol_file);
            load_params(params_file);
            mxnet::cpp::NDArray::WaitAll();

            dbg_default_trace("creating executor.");
            this->net.InferExecutorArrays(
                    global_ctx, &arg_arrays, &grad_arrays, &grad_reqs, &aux_arrays,
                    args_map, std::map<std::string, mxnet::cpp::NDArray>(),
                    std::map<std::string, mxnet::cpp::OpReqType>(), aux_map);
            for(auto& i : grad_reqs)
                i = mxnet::cpp::OpReqType::kNullOp;
            this->executor_pointer.reset(new mxnet::cpp::Executor(
                    net, global_ctx, arg_arrays, grad_arrays, grad_reqs, aux_arrays));
            dbg_default_trace("load_model() finished.");
			return true;
		} catch(const std::exception& e) {
            std::cerr << "Load model failed with exception " << e.what() << std::endl;
			return false;
        } catch(...) {
            std::cerr << "Load model failed with unknown exception." << std::endl;
            return false;
        }
    }

public:
    InferenceEngine(const mxnet::cpp::Context& ctxt,
                    const std::string& synset_file,
                    const std::string& symbol_file,
                    const std::string& params_file):
        global_ctx(ctxt),
        input_shape(std::vector<mxnet::cpp::index_t>({1, 3, 224, 224})) {
        dbg_default_trace("loading model begin.");
        load_model(synset_file, symbol_file, params_file);
        dbg_default_trace("loading model end.");
    }

    std::pair<std::string,double> infer(const ImageFrame& frame) {
        // do the inference.
#ifdef EVALUATION
        // uint64_t start_ns = get_time();
#endif
        // copy to input layer:
        FrameData *fd = reinterpret_cast<FrameData*>(frame.bytes);
        args_map["data"].SyncCopyFromCPU(reinterpret_cast<const mx_float*>(fd->data), input_shape.Size());
    
        this->executor_pointer->Forward(false);
        mxnet::cpp::NDArray::WaitAll();
        // extract the result
        auto output_shape = executor_pointer->outputs[0].GetShape();
        mxnet::cpp::NDArray output_in_cpu(output_shape,mxnet::cpp::Context::cpu());
        executor_pointer->outputs[0].CopyTo(&output_in_cpu);
        mxnet::cpp::NDArray::WaitAll();
        mx_float max = -1e10;
        int idx = -1;
        for(unsigned int jj = 0; jj < output_shape[1]; jj++) {
            if(max < output_in_cpu.At(0, jj)) {
                max = output_in_cpu.At(0, jj);
                idx = static_cast<int>(jj);
            }
        }
#ifdef EVALUATION
        // uint64_t end_ns = get_time();
#endif

        return {synset_vector[idx],max};
    }
};

class PetClassifierTrigger: public OffCriticalDataPathObserver {
private:
    // InferenceEngine flower_ie;
    // InferenceEngine pet_ie;
    mutable std::mutex p2p_send_mutex;
#ifdef EVALUATION
    int sock_fd;
    struct sockaddr_in serveraddr;
#endif
public:
    PetClassifierTrigger (): 
        OffCriticalDataPathObserver()// ,
        // flower_ie(derecho::getConfString(DPL_CONF_FLOWER_SYNSET),derecho::getConfString(DPL_CONF_FLOWER_SYMBOL),derecho::getConfString(DPL_CONF_FLOWER_PARAMS)),
        // pet_ie(derecho::getConfString(DPL_CONF_PET_SYNSET),derecho::getConfString(DPL_CONF_PET_SYMBOL),derecho::getConfString(DPL_CONF_PET_PARAMS))
        {
#ifdef EVALUATION
#define DPL_CONF_REPORT_TO	"CASCADE/report_to"
            uint16_t port;
            struct hostent *server;
            std::string hostname;
    		std::string report_to = derecho::getConfString(DPL_CONF_REPORT_TO);
            hostname = report_to.substr(0,report_to.find(":"));
            port = (uint16_t)std::stoi(report_to.substr(report_to.find(":")+1));
            sock_fd = socket(AF_INET, SOCK_DGRAM, 0);
            if (sock_fd < 0) {
                std::cerr << "Faile to open socket" << std::endl;
                return;
            }
            server = gethostbyname(hostname.c_str());
            if (server == nullptr) {
                std::cerr << "Failed to get host:" << hostname << std::endl;
            }

            bzero((char *) &serveraddr, sizeof(serveraddr));
            serveraddr.sin_family = AF_INET;
            bcopy((char *)server->h_addr, 
              (char *)&serveraddr.sin_addr.s_addr, server->h_length);
            serveraddr.sin_port = htons(port);
#endif
    }
    virtual void operator () (const std::string& key_string,
                              persistent::version_t version,
                              const mutils::ByteRepresentable* const value_ptr,
                              ICascadeContext* cascade_ctxt,
                              uint32_t worker_id) override {
        auto* ctxt = dynamic_cast<CascadeContext<VolatileCascadeStoreWithStringKey,PersistentCascadeStoreWithStringKey>*>(cascade_ctxt);
        /* step 1 prepare context */
        bool use_gpu = derecho::hasCustomizedConfKey(DPL_CONF_USE_GPU)?derecho::getConfBoolean(DPL_CONF_USE_GPU):false;
        if (use_gpu && ctxt->resource_descriptor.gpus.size()==0) {
            dbg_default_error("Worker{}: GPU is requested but no GPU found...giving up on processing data.",worker_id);
            return;
        }
        static thread_local const mxnet::cpp::Context mxnet_ctxt(
            use_gpu? mxnet::cpp::DeviceType::kGPU : mxnet::cpp::DeviceType::kCPU,
            use_gpu? ctxt->resource_descriptor.gpus[worker_id % ctxt->resource_descriptor.gpus.size()]:0);
        /* create inference engines */
        static thread_local InferenceEngine flower_ie(
                mxnet_ctxt,
                derecho::getConfString(DPL_CONF_FLOWER_SYNSET),
                derecho::getConfString(DPL_CONF_FLOWER_SYMBOL),
                derecho::getConfString(DPL_CONF_FLOWER_PARAMS));
        static thread_local InferenceEngine pet_ie(
                mxnet_ctxt,
                derecho::getConfString(DPL_CONF_PET_SYNSET),
                derecho::getConfString(DPL_CONF_PET_SYMBOL),
                derecho::getConfString(DPL_CONF_PET_PARAMS));
        size_t pos = key_string.rfind('/');
        std::string prefix;
        if (pos != std::string::npos) {
            prefix = key_string.substr(0,pos);
        }
        if (prefix == PET_PREFIX) {
            char* c;
            mutils::to_bytes(value_ptr,c);
            Blob b(c,mutils::bytes_size(c));
            ImageFrame f(key_string, b);
            ImageFrame* frame = &f;
            std::string name;
            double soft_max;
#ifdef EVALUATION
            uint64_t before_inference_ns = get_time();
#endif
            std::tie(name,soft_max) = pet_ie.infer(*frame);
#ifdef EVALUATION
            uint64_t after_inference_ns = get_time();
#endif
            PersistentCascadeStoreWithStringKey::ObjectType obj(frame->key,name.c_str(),name.size());
            std::lock_guard<std::mutex> lock(p2p_send_mutex);
#ifdef EVALUATION
            CloseLoopReport clr;
            FrameData* fd = reinterpret_cast<FrameData*>(frame->bytes);
            clr.photo_id = fd->photo_id;
            clr.inference_us = (after_inference_ns-before_inference_ns)/1000;
#endif
            auto result = ctxt->get_service_client_ref().template put<PersistentCascadeStoreWithStringKey>(obj);
            for (auto& reply_future:result.get()) {
                auto reply = reply_future.second.get();
                dbg_default_debug("node({}) replied with version:({:x},{}us)",reply_future.first,std::get<0>(reply),std::get<1>(reply));
#ifdef EVALUATION
            }
            uint64_t after_put_ns = get_time();
            clr.put_us = (after_put_ns-after_inference_ns)/1000;
            int serverlen = sizeof(serveraddr);
            size_t ns = sendto(sock_fd,(void*)&clr,sizeof(clr),0,(const sockaddr*)&serveraddr,serverlen);
            if (ns < 0) {
                std::cerr << "Failed to report error" << std::endl;
#endif
            }
        }
    }

    virtual ~PetClassifierTrigger() {
#ifdef EVALUATION
        close(sock_fd);
#endif
    }
};

class FlowerClassifierTrigger: public OffCriticalDataPathObserver {
private:
    // InferenceEngine flower_ie;
    // InferenceEngine pet_ie;
    mutable std::mutex p2p_send_mutex;
#ifdef EVALUATION
    int sock_fd;
    struct sockaddr_in serveraddr;
#endif
public:
    FlowerClassifierTrigger (): 
        OffCriticalDataPathObserver()// ,
        // flower_ie(derecho::getConfString(DPL_CONF_FLOWER_SYNSET),derecho::getConfString(DPL_CONF_FLOWER_SYMBOL),derecho::getConfString(DPL_CONF_FLOWER_PARAMS)),
        // pet_ie(derecho::getConfString(DPL_CONF_PET_SYNSET),derecho::getConfString(DPL_CONF_PET_SYMBOL),derecho::getConfString(DPL_CONF_PET_PARAMS))
        {
#ifdef EVALUATION
#define DPL_CONF_REPORT_TO	"CASCADE/report_to"
            uint16_t port;
            struct hostent *server;
            std::string hostname;
    		std::string report_to = derecho::getConfString(DPL_CONF_REPORT_TO);
            hostname = report_to.substr(0,report_to.find(":"));
            port = (uint16_t)std::stoi(report_to.substr(report_to.find(":")+1));
            sock_fd = socket(AF_INET, SOCK_DGRAM, 0);
            if (sock_fd < 0) {
                std::cerr << "Faile to open socket" << std::endl;
                return;
            }
            server = gethostbyname(hostname.c_str());
            if (server == nullptr) {
                std::cerr << "Failed to get host:" << hostname << std::endl;
            }

            bzero((char *) &serveraddr, sizeof(serveraddr));
            serveraddr.sin_family = AF_INET;
            bcopy((char *)server->h_addr, 
              (char *)&serveraddr.sin_addr.s_addr, server->h_length);
            serveraddr.sin_port = htons(port);
#endif
    }

    virtual void operator () (const std::string& key_string,
                              persistent::version_t version,
                              const mutils::ByteRepresentable* const value_ptr,
                              ICascadeContext* cascade_ctxt,
                              uint32_t worker_id) override {
        auto* ctxt = dynamic_cast<CascadeContext<VolatileCascadeStoreWithStringKey,PersistentCascadeStoreWithStringKey>*>(cascade_ctxt);
        /* step 1 prepare context */
        bool use_gpu = derecho::hasCustomizedConfKey(DPL_CONF_USE_GPU)?derecho::getConfBoolean(DPL_CONF_USE_GPU):false;
        if (use_gpu && ctxt->resource_descriptor.gpus.size()==0) {
            dbg_default_error("Worker{}: GPU is requested but no GPU found...giving up on processing data.",worker_id);
            return;
        }
        static thread_local const mxnet::cpp::Context mxnet_ctxt(
            use_gpu? mxnet::cpp::DeviceType::kGPU : mxnet::cpp::DeviceType::kCPU,
            use_gpu? ctxt->resource_descriptor.gpus[worker_id % ctxt->resource_descriptor.gpus.size()]:0);
        /* create inference engines */
        static thread_local InferenceEngine flower_ie(
                mxnet_ctxt,
                derecho::getConfString(DPL_CONF_FLOWER_SYNSET),
                derecho::getConfString(DPL_CONF_FLOWER_SYMBOL),
                derecho::getConfString(DPL_CONF_FLOWER_PARAMS));
        static thread_local InferenceEngine pet_ie(
                mxnet_ctxt,
                derecho::getConfString(DPL_CONF_PET_SYNSET),
                derecho::getConfString(DPL_CONF_PET_SYMBOL),
                derecho::getConfString(DPL_CONF_PET_PARAMS));
        size_t pos = key_string.rfind('/');
        std::string prefix;
        if (pos != std::string::npos) {
            prefix = key_string.substr(0,pos);
        }
        if (prefix == FLOWER_PREFIX) {
            char* c;
            mutils::to_bytes(value_ptr,c);
            Blob b(c,mutils::bytes_size(c));
            ImageFrame f(key_string, b);
            ImageFrame* frame = &f;
            std::string name;
            double soft_max;
#ifdef EVALUATION
            uint64_t before_inference_ns = get_time();
#endif
            std::tie(name,soft_max) = pet_ie.infer(*frame);
#ifdef EVALUATION
            uint64_t after_inference_ns = get_time();
#endif
            PersistentCascadeStoreWithStringKey::ObjectType obj(frame->key,name.c_str(),name.size());
            std::lock_guard<std::mutex> lock(p2p_send_mutex);
#ifdef EVALUATION
            CloseLoopReport clr;
            FrameData* fd = reinterpret_cast<FrameData*>(frame->bytes);
            clr.photo_id = fd->photo_id;
            clr.inference_us = (after_inference_ns-before_inference_ns)/1000;
#endif
            auto result = ctxt->get_service_client_ref().template put<PersistentCascadeStoreWithStringKey>(obj);
            for (auto& reply_future:result.get()) {
                auto reply = reply_future.second.get();
                dbg_default_debug("node({}) replied with version:({:x},{}us)",reply_future.first,std::get<0>(reply),std::get<1>(reply));
#ifdef EVALUATION
            }
            uint64_t after_put_ns = get_time();
            clr.put_us = (after_put_ns-after_inference_ns)/1000;
            int serverlen = sizeof(serveraddr);
            size_t ns = sendto(sock_fd,(void*)&clr,sizeof(clr),0,(const sockaddr*)&serveraddr,serverlen);
            if (ns < 0) {
                std::cerr << "Failed to report error" << std::endl;
#endif
            }
        }
    }

    virtual ~FlowerClassifierTrigger() {
#ifdef EVALUATION
        close(sock_fd);
#endif
    }
};

void register_triggers(ICascadeContext* ctxt) {
    auto* typed_ctxt = dynamic_cast<CascadeContext<VolatileCascadeStoreWithStringKey,PersistentCascadeStoreWithStringKey>*>(ctxt);
    typed_ctxt->register_prefixes({PET_PREFIX},MY_UUID,std::make_shared<PetClassifierTrigger>());
    typed_ctxt->register_prefixes({FLOWER_PREFIX},MY_UUID,std::make_shared<FlowerClassifierTrigger>());
}

void unregister_triggers(ICascadeContext* ctxt) {
    auto* typed_ctxt = dynamic_cast<CascadeContext<VolatileCascadeStoreWithStringKey,PersistentCascadeStoreWithStringKey>*>(ctxt);
    typed_ctxt->unregister_prefixes({PET_PREFIX,FLOWER_PREFIX},MY_UUID);
}

} // namespace cascade
} // namespace derecho
