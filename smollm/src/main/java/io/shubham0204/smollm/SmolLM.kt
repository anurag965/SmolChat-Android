package io.shubham0204.smollm

import android.os.Build
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileNotFoundException
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/** This class interacts with the JNI binding and provides a Kotlin API to infer a GGUF LLM model */
class SmolLM {
    companion object {
        private const val i8mmEnabled = true

        init {
            val logTag = SmolLM::class.java.simpleName

            // check if the following CPU features are available,
            // and load the native library accordingly
            val cpuFeatures = getCPUFeatures()
            val hasFp16 = cpuFeatures.contains("fp16") || cpuFeatures.contains("fphp")
            val hasDotProd = cpuFeatures.contains("dotprod") || cpuFeatures.contains("asimddp")
            val hasSve = cpuFeatures.contains("sve")
            val hasI8mm = cpuFeatures.contains("i8mm")
            val isAtLeastArmV82 =
                cpuFeatures.contains("asimd") &&
                    cpuFeatures.contains("crc32") &&
                        cpuFeatures.contains("aes")
            val isAtLeastArmV84 = cpuFeatures.contains("dcpop") && cpuFeatures.contains("uscat")

            Log.d(logTag, "CPU features: $cpuFeatures")
            Log.d(logTag, "- hasFp16: $hasFp16")
            Log.d(logTag, "- hasDotProd: $hasDotProd")
            Log.d(logTag, "- hasSve: $hasSve")
            Log.d(logTag, "- hasI8mm: $hasI8mm")
            Log.d(logTag, "- isAtLeastArmV82: $isAtLeastArmV82")
            Log.d(logTag, "- isAtLeastArmV84: $isAtLeastArmV84")

            // Check if the app is running in an emulated device
            val isEmulated =
                (Build.HARDWARE.contains("goldfish") || Build.HARDWARE.contains("ranchu"))
            Log.d(logTag, "isEmulated: $isEmulated")

            if (!isEmulated) {
                if (supportsArm64V8a()) {
                    if (isAtLeastArmV84 && hasSve && hasI8mm && hasFp16 && hasDotProd) {
                        Log.d(logTag, "Loading libsmollm_v8_4_fp16_dotprod_i8mm_sve.so")
                        System.loadLibrary("smollm_v8_4_fp16_dotprod_i8mm_sve")
                    } else if (isAtLeastArmV84 && hasSve && hasFp16 && hasDotProd) {
                        Log.d(logTag, "Loading libsmollm_v8_4_fp16_dotprod_sve.so")
                        System.loadLibrary("smollm_v8_4_fp16_dotprod_sve")
                    } else if (isAtLeastArmV84 && i8mmEnabled && hasI8mm && hasFp16 && hasDotProd) {
                        Log.d(logTag, "Loading libsmollm_v8_4_fp16_dotprod_i8mm.so")
                        System.loadLibrary("smollm_v8_4_fp16_dotprod_i8mm")
                    } else if (isAtLeastArmV84 && hasFp16 && hasDotProd) {
                        Log.d(logTag, "Loading libsmollm_v8_4_fp16_dotprod.so")
                        System.loadLibrary("smollm_v8_4_fp16_dotprod")
                    } else if (isAtLeastArmV82 && hasFp16 && hasDotProd) {
                        Log.d(logTag, "Loading libsmollm_v8_2_fp16_dotprod.so")
                        System.loadLibrary("smollm_v8_2_fp16_dotprod")
                    } else if (isAtLeastArmV82 && hasFp16) {
                        Log.d(logTag, "Loading libsmollm_v8_2_fp16.so")
                        System.loadLibrary("smollm_v8_2_fp16")
                    } else {
                        Log.d(logTag, "Loading libsmollm_v8.so")
                        System.loadLibrary("smollm_v8")
                    }
                } else if (Build.SUPPORTED_32_BIT_ABIS[0]?.equals("armeabi-v7a") == true) {
                    // armv7a (32bit) device
                    Log.d(logTag, "Loading libsmollm_v7a.so")
                    System.loadLibrary("smollm_v7a")
                } else {
                    Log.d(logTag, "Loading default libsmollm.so")
                    System.loadLibrary("smollm")
                }
            } else {
                Log.d(logTag, "Loading default libsmollm.so")
                System.loadLibrary("smollm")
            }
        }

        /**
         * Reads the /proc/cpuinfo file and returns the line starting with 'Features :' that
         * containing the available CPU features
         */
        private fun getCPUFeatures(): String {
            val cpuInfo =
                try {
                    File("/proc/cpuinfo").readText()
                } catch (e: FileNotFoundException) {
                    ""
                }
            val cpuFeatures =
                cpuInfo.substringAfter("Features").substringAfter(":").substringBefore("\n").trim()
            return cpuFeatures
        }

        private fun supportsArm64V8a(): Boolean = Build.SUPPORTED_ABIS[0].equals("arm64-v8a")
    }

    // Native LLMInference pointer
    private var nativePtr = 0L
    private val ptrLock = ReentrantLock()

    /**
     * Provides default values for inference parameters. These values are used when the
     * corresponding parameters are not provided by the user or are not available in the GGUF model
     * file.
     */
    object DefaultInferenceParams {
        val contextSize: Long = 1024L
        val chatTemplate: String =
            "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|> ' }}{% endif %}{{'<|im_start|>' + message['role'] + ' ' + message['content'] + '<|im_end|>' + ' '}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant ' }}{% endif %}"
    }

    data class InferenceParams(
        val minP: Float = 0.1f,
        val temperature: Float = 0.8f,
        val storeChats: Boolean = true,
        val contextSize: Long? = null,
        val chatTemplate: String? = null,
        val numThreads: Int = 4,
        val useMmap: Boolean = true,
        val useMlock: Boolean = false,
    )

    suspend fun load(modelPath: String, params: InferenceParams = InferenceParams()) =
        withContext(Dispatchers.IO) {
            val ggufReader = GGUFReader()
            ggufReader.load(modelPath)
            val modelContextSize = ggufReader.getContextSize() ?: DefaultInferenceParams.contextSize
            val modelChatTemplate =
                ggufReader.getChatTemplate() ?: DefaultInferenceParams.chatTemplate
            
            ptrLock.withLock {
                if (nativePtr != 0L) {
                    close(nativePtr)
                    nativePtr = 0L
                }
                nativePtr =
                    loadModel(
                        modelPath,
                        params.minP,
                        params.temperature,
                        params.storeChats,
                        params.contextSize ?: modelContextSize,
                        params.chatTemplate ?: modelChatTemplate,
                        params.numThreads,
                        params.useMmap,
                        params.useMlock,
                    )
            }
        }

    fun addUserMessage(message: String) = ptrLock.withLock {
        verifyHandle()
        addChatMessage(nativePtr, message, "user")
    }

    fun addSystemPrompt(prompt: String) = ptrLock.withLock {
        verifyHandle()
        addChatMessage(nativePtr, prompt, "system")
    }

    fun addAssistantMessage(message: String) = ptrLock.withLock {
        verifyHandle()
        addChatMessage(nativePtr, message, "assistant")
    }

    fun getResponseGenerationSpeed(): Float = ptrLock.withLock {
        verifyHandle()
        return getResponseGenerationSpeed(nativePtr)
    }

    fun getContextLengthUsed(): Int = ptrLock.withLock {
        verifyHandle()
        return getContextSizeUsed(nativePtr)
    }

    fun getResponseAsFlow(query: String): Flow<String> = flow {
        ptrLock.withLock {
            verifyHandle()
            startCompletion(nativePtr, query)
        }
        while (true) {
            val piece = ptrLock.withLock {
                completionLoop(nativePtr)
            }
            if (piece == "[EOG]") break
            emit(piece)
        }
        ptrLock.withLock {
            stopCompletion(nativePtr)
        }
    }

    /**
     * Multimodal version of getResponseAsFlow.
     * Skips startCompletion as buildVideoChat already evaluated the prompt.
     */
    fun getMultimodalResponseAsFlow(): Flow<String> = flow {
        ptrLock.withLock {
            verifyHandle()
        }
        while (true) {
            val piece = ptrLock.withLock {
                completionLoop(nativePtr)
            }
            if (piece == "[EOG]") break
            emit(piece)
        }
        ptrLock.withLock {
            stopCompletion(nativePtr)
        }
    }

    fun getResponse(query: String): String = ptrLock.withLock {
        verifyHandle()
        startCompletion(nativePtr, query)
        var response = ""
        while (true) {
            val piece = completionLoop(nativePtr)
            if (piece == "[EOG]") break
            response += piece
        }
        stopCompletion(nativePtr)
        return response
    }

    fun close() {
        ptrLock.withLock {
            if (nativePtr != 0L) {
                close(nativePtr)
                nativePtr = 0L
            }
        }
    }

    private fun verifyHandle() {
        assert(nativePtr != 0L) { "Model is not loaded. Use SmolLM.load to load the model" }
    }

    // ========== EXISTING TEXT JNI ==========

    private external fun loadModel(
        modelPath: String,
        minP: Float,
        temperature: Float,
        storeChats: Boolean,
        contextSize: Long,
        chatTemplate: String,
        nThreads: Int,
        useMmap: Boolean,
        useMlock: Boolean,
    ): Long

    private external fun addChatMessage(modelPtr: Long, message: String, role: String)

    private external fun getResponseGenerationSpeed(modelPtr: Long): Float

    private external fun getContextSizeUsed(modelPtr: Long): Int

    private external fun close(modelPtr: Long)

    private external fun startCompletion(modelPtr: Long, prompt: String)

    private external fun completionLoop(modelPtr: Long): String

    private external fun stopCompletion(modelPtr: Long)

    // ========== NEW MULTIMODAL / VIDEO JNI ==========

    private external fun loadMultimodalModel(
        modelPath: String,
        mmprojPath: String,
        minP: Float,
        temperature: Float,
        nGpuLayers: Int,
        contextSize: Long,
    ): Long

    private external fun addVideoFrame(
        modelPtr: Long,
        data: ByteArray,
        width: Int,
        height: Int,
        channels: Int,
    )

    private external fun buildMultimodalChat(
        modelPtr: Long,
        prompt: String,
    ): Boolean

    private external fun clearVideoFrames(
        modelPtr: Long,
    )

    private external fun getFrameCount(
        modelPtr: Long,
    ): Int

    // ========== PUBLIC VIDEO / SmolVLM2 API ==========

    /**
     * Load SmolVLM2-500M-Video-Instruct + mmproj (video model).
     */
    suspend fun loadVideoModel(
        modelPath: String,
        mmprojPath: String,
        minP: Float = 0.05f,
        temperature: Float = 0.2f,
        nGpuLayers: Int = 35,
    ) = withContext(Dispatchers.IO) {
        val ggufReader = GGUFReader()
        ggufReader.load(modelPath)
        val modelContextSize = ggufReader.getContextSize() ?: DefaultInferenceParams.contextSize
        
        ptrLock.withLock {
            if (nativePtr != 0L) {
                close(nativePtr)
                nativePtr = 0L
            }
            nativePtr = loadMultimodalModel(modelPath, mmprojPath, minP, temperature, nGpuLayers, modelContextSize)
        }
    }

    /**
     * Add one RGB frame (width x height x 3) to the VLM.
     */
    fun addVideoFrameRGB(
        rgbData: ByteArray,
        width: Int,
        height: Int,
    ) {
        ptrLock.withLock {
            verifyHandle()
            addVideoFrame(nativePtr, rgbData, width, height, 3)
        }
    }

    /**
     * Build multimodal chat: all added frames + prompt.
     */
    fun buildVideoChat(prompt: String): Boolean = ptrLock.withLock {
        verifyHandle()
        return buildMultimodalChat(nativePtr, prompt)
    }

    /**
     * Clear buffered frames.
     */
    fun clearFrames() {
        ptrLock.withLock {
            if (nativePtr != 0L) {
                clearVideoFrames(nativePtr)
            }
        }
    }

    fun frameCount(): Int = ptrLock.withLock {
        return if (nativePtr != 0L) getFrameCount(nativePtr) else 0
    }
}
