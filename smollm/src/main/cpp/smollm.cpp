#include "LLMInference.h"
#include <jni.h>
#include <android/log.h>

#define TAG "[SmolLM-JNI]"
#define LOGi(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)

extern "C" JNIEXPORT jlong JNICALL
Java_io_shubham0204_smollm_SmolLM_loadModel(JNIEnv* env, jobject thiz, jstring modelPath, jfloat minP,
                                            jfloat temperature, jboolean storeChats, jlong contextSize,
                                            jstring chatTemplate, jint nThreads, jboolean useMmap, jboolean useMlock) {
    jboolean    isCopy           = true;
    const char* modelPathCstr    = env->GetStringUTFChars(modelPath, &isCopy);
    auto*       llmInference     = new LLMInference();
    const char* chatTemplateCstr = env->GetStringUTFChars(chatTemplate, &isCopy);

    try {
        llmInference->loadModel(modelPathCstr, minP, temperature, storeChats, contextSize, chatTemplateCstr, nThreads,
                                useMmap, useMlock);
    } catch (std::runtime_error& error) {
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), error.what());
    }

    env->ReleaseStringUTFChars(modelPath, modelPathCstr);
    env->ReleaseStringUTFChars(chatTemplate, chatTemplateCstr);
    return reinterpret_cast<jlong>(llmInference);
}

extern "C" JNIEXPORT void JNICALL
Java_io_shubham0204_smollm_SmolLM_addChatMessage(JNIEnv* env, jobject thiz, jlong modelPtr, jstring message,
                                                 jstring role) {
    jboolean    isCopy       = true;
    const char* messageCstr  = env->GetStringUTFChars(message, &isCopy);
    const char* roleCstr     = env->GetStringUTFChars(role, &isCopy);
    auto*       llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    llmInference->addChatMessage(messageCstr, roleCstr);
    env->ReleaseStringUTFChars(message, messageCstr);
    env->ReleaseStringUTFChars(role, roleCstr);
}

extern "C" JNIEXPORT jfloat JNICALL
Java_io_shubham0204_smollm_SmolLM_getResponseGenerationSpeed(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    return llmInference->getResponseGenerationTime();
}

extern "C" JNIEXPORT jint JNICALL
Java_io_shubham0204_smollm_SmolLM_getContextSizeUsed(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    return llmInference->getContextSizeUsed();
}

extern "C" JNIEXPORT void JNICALL
Java_io_shubham0204_smollm_SmolLM_close(JNIEnv* env, jobject thiz, jlong modelPtr) {
    LOGi("close, modelPtr: %ld", modelPtr);
    auto* llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    delete llmInference;
}

extern "C" JNIEXPORT void JNICALL
Java_io_shubham0204_smollm_SmolLM_startCompletion(JNIEnv* env, jobject thiz, jlong modelPtr, jstring prompt) {
    jboolean    isCopy       = true;
    const char* promptCstr   = env->GetStringUTFChars(prompt, &isCopy);
    auto*       llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    llmInference->startCompletion(promptCstr);
    env->ReleaseStringUTFChars(prompt, promptCstr);
}

extern "C" JNIEXPORT jstring JNICALL
Java_io_shubham0204_smollm_SmolLM_completionLoop(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    try {
        std::string response = llmInference->completionLoop();
        return env->NewStringUTF(response.c_str());
    } catch (std::runtime_error& error) {
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), error.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_io_shubham0204_smollm_SmolLM_stopCompletion(JNIEnv* env, jobject thiz, jlong modelPtr) {
    auto* llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    llmInference->stopCompletion();
}

// ========== MULTIMODAL / VIDEO JNI BRIDGE (NEW) ==========

extern "C" JNIEXPORT jlong JNICALL
Java_io_shubham0204_smollm_SmolLM_loadMultimodalModel(JNIEnv* env,
                                                      jobject thiz,
                                                      jstring modelPath,
                                                      jstring mmprojPath,
                                                      jfloat minP,
                                                      jfloat temperature,
                                                      jint nGpuLayers,
                                                      jlong contextSize) {
    jboolean    isCopy        = true;
    const char* modelPathCstr = env->GetStringUTFChars(modelPath, &isCopy);
    const char* mmprojCstr    = env->GetStringUTFChars(mmprojPath, &isCopy);

    auto* llmInference = new LLMInference();

    bool ok = llmInference->loadMultimodalModel(modelPathCstr, mmprojCstr, (float) minP, (float) temperature, (int) nGpuLayers, (long) contextSize);
    if (!ok) {
        env->ReleaseStringUTFChars(modelPath,  modelPathCstr);
        env->ReleaseStringUTFChars(mmprojPath, mmprojCstr);
        delete llmInference;
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),
                      "Failed to load multimodal model or mmproj");
        return 0;
    }

    env->ReleaseStringUTFChars(modelPath,  modelPathCstr);
    env->ReleaseStringUTFChars(mmprojPath, mmprojCstr);

    return reinterpret_cast<jlong>(llmInference);
}

extern "C" JNIEXPORT void JNICALL
Java_io_shubham0204_smollm_SmolLM_addVideoFrame(JNIEnv* env,
                                                jobject thiz,
                                                jlong modelPtr,
                                                jbyteArray data,
                                                jint width,
                                                jint height,
                                                jint channels) {
    LOGi("addVideoFrame, modelPtr: %ld", modelPtr);
    auto* llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    if (!llmInference) return;

    jsize len = env->GetArrayLength(data);
    if (len != width * height * channels) {
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),
                      "Pixel data size does not match dimensions");
        return;
    }

    jbyte* bytes = env->GetByteArrayElements(data, nullptr);

    llmInference->addVideoFrame(reinterpret_cast<uint8_t*>(bytes),
                                (int) width, (int) height, (int) channels);

    env->ReleaseByteArrayElements(data, bytes, JNI_ABORT);
}

extern "C" JNIEXPORT jboolean JNICALL
Java_io_shubham0204_smollm_SmolLM_buildMultimodalChat(JNIEnv* env,
                                                      jobject thiz,
                                                      jlong modelPtr,
                                                      jstring prompt) {
    LOGi("buildMultimodalChat, modelPtr: %ld", modelPtr);
    auto* llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    if (!llmInference) return JNI_FALSE;

    jboolean    isCopy     = true;
    const char* promptCstr = env->GetStringUTFChars(prompt, &isCopy);

    bool ok = llmInference->buildMultimodalChat(promptCstr);

    env->ReleaseStringUTFChars(prompt, promptCstr);

    return ok ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_io_shubham0204_smollm_SmolLM_clearVideoFrames(JNIEnv* env,
                                                   jobject thiz,
                                                   jlong modelPtr) {
    LOGi("clearVideoFrames, modelPtr: %ld", modelPtr);
    auto* llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    if (!llmInference) return;
    llmInference->clearVideoFrames();
}

extern "C" JNIEXPORT jint JNICALL
Java_io_shubham0204_smollm_SmolLM_getFrameCount(JNIEnv* env,
                                                jobject thiz,
                                                jlong modelPtr) {
    auto* llmInference = reinterpret_cast<LLMInference*>(modelPtr);
    if (!llmInference) return 0;
    return (jint) llmInference->getFrameCount();
}
