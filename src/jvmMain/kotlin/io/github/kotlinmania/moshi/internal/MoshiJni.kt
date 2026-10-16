// port-lint: source N/A
package io.github.kotlinmania.moshi.internal

internal object MoshiJni {
    init {
        System.loadLibrary("moshi_jni")
    }

    @JvmStatic
    external fun getpid(): Int
}
