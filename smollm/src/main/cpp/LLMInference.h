#pragma once
#include "llama.h"
#include "common.h"
#include "mtmd.h"
#include <string>
#include <vector>

class LLMInference {
    // llama.cpp-specific types
    llama_context* _ctx = nullptr;
    llama_model*   _model = nullptr;
    llama_sampler* _sampler = nullptr;
    llama_token    _currToken = 0;
    llama_batch*   _batch = nullptr;
    mtmd_context*  _mtmd_ctx = nullptr;

    // container to store user/assistant messages in the chat
    std::vector<llama_chat_message> _messages;
    // stores the string generated after applying
    // the chat-template to all messages in `_messages`
    std::vector<char> _formattedMessages;
    // stores the tokens for the last query
    // appended to `_messages`
    std::vector<llama_token> _promptTokens;
    int                      _prevLen = 0;
    const char*              _chatTemplate = nullptr;

    // stores the complete response for the given query
    std::string _response;
    std::string _cacheResponseTokens;
    // whether to cache previous messages in `_messages`
    bool _storeChats = false;

    // response generation metrics
    int64_t _responseGenerationTime = 0;
    long    _responseNumTokens      = 0;

    // length of context window consumed during the conversation
    int _nCtxUsed = 0;

    bool _isValidUtf8(const char* response);

    // ========== VIDEO CAPTIONING (NEW) ==========
private:
    struct ImageFrame {
        std::vector<uint8_t> data;
        int width, height, channels;
    };
    std::vector<ImageFrame> videoFrames;
    std::string mmproj_path;
    bool is_multimodal_model = false;

public:
    // ========== EXISTING METHODS ==========
    void loadModel(const char* modelPath, float minP, float temperature, bool storeChats, long contextSize,
                   const char* chatTemplate, int nThreads, bool useMmap, bool useMlock);

    void addChatMessage(const char* message, const char* role);

    float getResponseGenerationTime() const;

    int getContextSizeUsed() const;

    void startCompletion(const char* query);

    std::string completionLoop();

    void stopCompletion();

    // ========== NEW VIDEO METHODS ==========
    bool loadMultimodalModel(const char* model_path, const char* mmproj_path, float minP = 0.05f, float temperature = 0.2f, int n_gpu_layers = 35, long contextSize = 4096);
    void addVideoFrame(const uint8_t* pixel_data, int width, int height, int channels);
    bool buildMultimodalChat(const char* text_prompt);
    void clearVideoFrames();
    int getFrameCount() const;

    ~LLMInference();
};
