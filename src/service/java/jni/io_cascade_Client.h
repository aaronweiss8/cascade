/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class io_cascade_Client */

#ifndef _Included_io_cascade_Client
#define _Included_io_cascade_Client
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     io_cascade_Client
 * Method:    createClient
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_io_cascade_Client_createClient
  (JNIEnv *, jobject);

/*
 * Class:     io_cascade_Client
 * Method:    getMembers
 * Signature: ()Ljava/util/List;
 */
JNIEXPORT jobject JNICALL Java_io_cascade_Client_getMembers
  (JNIEnv *, jobject);

/*
 * Class:     io_cascade_Client
 * Method:    getShardMembers
 * Signature: (Lio/cascade/ServiceType;JJ)Ljava/util/List;
 */
JNIEXPORT jobject JNICALL Java_io_cascade_Client_getShardMembers
  (JNIEnv *, jobject, jobject, jlong, jlong);

/*
 * Class:     io_cascade_Client
 * Method:    setMemberSelectionPolicy
 * Signature: (Lio/cascade/ServiceType;JJLio/cascade/ShardMemberSelectionPolicy;)V
 */
JNIEXPORT void JNICALL Java_io_cascade_Client_setMemberSelectionPolicy
  (JNIEnv *, jobject, jobject, jlong, jlong, jobject);

/*
 * Class:     io_cascade_Client
 * Method:    getMemberSelectionPolicy
 * Signature: (Lio/cascade/ServiceType;JJ)Lio/cascade/ShardMemberSelectionPolicy;
 */
JNIEXPORT jobject JNICALL Java_io_cascade_Client_getMemberSelectionPolicy
  (JNIEnv *, jobject, jobject, jlong, jlong);

/*
 * Class:     io_cascade_Client
 * Method:    putInternal
 * Signature: (Lio/cascade/ServiceType;JJLjava/nio/ByteBuffer;Ljava/nio/ByteBuffer;)J
 */
JNIEXPORT jlong JNICALL Java_io_cascade_Client_putInternal
  (JNIEnv *, jobject, jobject, jlong, jlong, jobject, jobject);

/*
 * Class:     io_cascade_Client
 * Method:    getInternal
 * Signature: (Lio/cascade/ServiceType;JJLjava/nio/ByteBuffer;J)J
 */
JNIEXPORT jlong JNICALL Java_io_cascade_Client_getInternal
  (JNIEnv *, jobject, jobject, jlong, jlong, jobject, jlong);

/*
 * Class:     io_cascade_Client
 * Method:    getInternalByTime
 * Signature: (Lio/cascade/ServiceType;JJLjava/nio/ByteBuffer;J)J
 */
JNIEXPORT jlong JNICALL Java_io_cascade_Client_getInternalByTime
  (JNIEnv *, jobject, jobject, jlong, jlong, jobject, jlong);

/*
 * Class:     io_cascade_Client
 * Method:    removeInternal
 * Signature: (Lio/cascade/ServiceType;JJLjava/nio/ByteBuffer;)J
 */
JNIEXPORT jlong JNICALL Java_io_cascade_Client_removeInternal
  (JNIEnv *, jobject, jobject, jlong, jlong, jobject);

#ifdef __cplusplus
}
#endif
#endif
