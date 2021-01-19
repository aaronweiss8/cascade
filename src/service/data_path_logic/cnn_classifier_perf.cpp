#include <cascade/service_client_api.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sstream>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include "cnn_classifier_dpl_eval.hpp"

using namespace derecho::cascade;

VolatileCascadeStoreWithStringKey::ObjectType get_photo_object(const char* type, const char* key, const char* photo_file) {
    int fd;
    struct stat st;
    void* file_data;

    // open and map file
    if(stat(photo_file, &st) || access(photo_file, R_OK)) {
		std::cerr << "file " << photo_file << " is not readable." << std::endl;
		return VolatileCascadeStoreWithStringKey::ObjectType::IV;
    }

    if((S_IFMT & st.st_mode) != S_IFREG) {
		std::cerr << photo_file << " is not a regular file." << std::endl;
		return VolatileCascadeStoreWithStringKey::ObjectType::IV;
    }

    if((fd = open(photo_file, O_RDONLY)) < 0) {
        std::cerr << "Failed to open file(" << photo_file << ") in readonly mode with "
                  << "error:" << strerror(errno) << "." << std::endl;
        return VolatileCascadeStoreWithStringKey::ObjectType::IV;
    }

    if((file_data = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE,
                         fd, 0))
       == MAP_FAILED) {
        std::cerr << "Failed to map file(" << photo_file << ") with "
                  << "error:" << strerror(errno) << "." << std::endl;
        return VolatileCascadeStoreWithStringKey::ObjectType::IV;
    }
    
    // create Object
    return VolatileCascadeStoreWithStringKey::ObjectType(std::string(type)+"/"+key,static_cast<const char*>(file_data),st.st_size);
}

auto parse_file_list(const char* type, const char* files) {
    std::vector<VolatileCascadeStoreWithStringKey::ObjectType> vec;
    std::istringstream fs(files);
    std::string f;
    size_t key = 0;
    while(std::getline(fs,f,':')){
        vec.push_back(get_photo_object(type,std::to_string(key).c_str(),f.c_str()));
        key ++;
    }
    return vec;
}

#ifdef EVALUATION
#define BUFSIZE (256)
void collect_time(uint16_t udp_port, size_t num_photos, size_t num_messages, uint64_t timestamps[]) {
    struct sockaddr_in serveraddr,clientaddr;
    char buf[BUFSIZE];
    //STEP 1: start UDP channel
  	int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  	if (sockfd < 0) {
        std::cerr << "ERROR opening socket" << std::endl;
        return;
    }
  	int optval = 1;
  	setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (const void *)&optval , sizeof(int));
    //STEP 2: waiting for UDP message
    bzero((char *) &serveraddr, sizeof(serveraddr));
    serveraddr.sin_family = AF_INET;
    serveraddr.sin_addr.s_addr = htonl(INADDR_ANY);
    serveraddr.sin_port = htons(udp_port);
    if (bind(sockfd, (struct sockaddr *) &serveraddr, sizeof(serveraddr)) < 0) {
        std::cerr << "Fail to bind udp port:" << udp_port << std::endl;
    	return;
    }
    socklen_t clientlen = sizeof(clientaddr);
    size_t cnt = 0;
    while (cnt < num_messages) {
        ssize_t nrecv = recvfrom(sockfd,buf,BUFSIZE,0,(struct sockaddr *) &clientaddr,&clientlen);
        if (nrecv < 0) {
            std::cerr << "Fail to recv udp package." << std::endl;
            return;
        }
        CloseLoopReport *clr = reinterpret_cast<CloseLoopReport*>(buf);
        timestamps[clr->ver*num_photos + clr->key] = get_time();
        cnt ++;
    }
    //STEP 3: finish 
    close(sockfd);
}
#endif

/**
 * The cnn classifier client post photos to cascade to be processed by the cnn classifier data path logic.
 */
int main(int argc, char** argv) {
    const char* HELP_INFO = "--(t)ype <pet|flower> --(f)iles <file1:file2:file3...>\n"
                            "--(n)um_messages <number of messages, default to 100>\n"
#ifdef EVALUATION
                            "--(u)dp_port <UDP port for report server. For evaluation only, default=54321>\n"
#endif
                            "--(h)elp";
    int c;
    static struct option long_options[] = {
        {"files",   required_argument,  0,  'f'},
        {"type",    required_argument,  0,  't'},
        {"num_messages",    required_argument,  0,  'n'},
#ifdef EVALUATION
        {"udp_port",		required_argument,  0,  'u'},
#endif
        {"help",    no_argument,        0,  'h'},
        {0,0,0,0}
    };
    const char* files = nullptr;
    const char* type = nullptr;
    size_t num_messages = 100;
#ifdef EVALUATION
    uint16_t udp_port = 54321;
#endif
    bool print_help = false;

    while(true){
        int option_index = 0;
        c = getopt_long(argc,argv,"f:t:n:h",long_options,&option_index);
        if (c == -1) {
            break;
        }

        switch (c) {
        case 'f':
            files = optarg;
            break;
        case 't':
            type = optarg;
            break;
        case 'n':
            num_messages = std::stol(optarg);
            break;
#ifdef EVALUATION
        case 'u':
            udp_port = std::stol(optarg);
            break;
#endif
        case 'h':
            print_help = true;
            std::cout << "Usage: " << argv[0] << " " << HELP_INFO << std::endl;
            break;
        }
    }

    if (!(files && type)) {
        if (!print_help) {
            std::cout << "Invalid argument." << std::endl;
            std::cout << "Usage: " << argv[0] << " " << HELP_INFO << std::endl;
            return -1;
        }
    } else {
        auto vec_photos = parse_file_list(type,files);
        const size_t vec_size = vec_photos.size();
        ServiceClientAPI capi;
        // TODO: change this to asynchronous send.
#ifdef EVALUATION
        uint64_t send_message_ts[num_messages];
        uint64_t close_loop_ts[num_messages];
#endif
        for(size_t i=0;i<num_messages;i++) {
            /*
             * key = [i%vec_size]
             * version = i/vec_size
             * i = stoi(key) + version*vec_size;
             */
            auto ret = capi.template put<VolatileCascadeStoreWithStringKey>(vec_photos.at(i%vec_size), 0, 0);
#ifdef EVALUATION
            send_message_ts[i] = get_time();
#endif
            for (auto& reply_future:ret.get()) {
                auto reply = reply_future.second.get();
                std::cout << "node(" << reply_future.first << ") replied with version:" << std::get<0>(reply)
                          << ",ts_us:" << std::get<1>(reply) << std::endl;
            }
        }
#ifdef EVALUATION
        collect_time(udp_port, vec_size, num_messages, close_loop_ts);
        // TODO: evaluate using the data
        std::cout << "Timespan:" << (close_loop_ts - send_message_ts[0]) << "us" << std::endl;
#endif
    }

    return 0;
}
