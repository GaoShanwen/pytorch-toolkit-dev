#include "encryptor.h"
#include <openssl/aes.h>
#include <memory.h>

// #include <iostream>
// #include <fstream>
// #include <stdio.h>
// #include <string.h>
// #include <vector>
// #include <sstream>
// #include <openssl/err.h>
// #include <iomanip>
#include "spdlog/spdlog.h"
#include <spdlog/cfg/env.h>  // support for loading levels from the environment  variable
#include <spdlog/fmt/ostr.h> // support for user defined types

using namespace spdlog;
using namespace std;

Encryptor::Encryptor(){
	init();
    //spdlog::info("Encryptor: {}\n", "init4");
}


Encryptor::~Encryptor(){
	release();
    //spdlog::info("Encryptor: {}\n", "init5");
}


bool Encryptor::init(){
	memset(m_pszAESKeyBuf, 0, sizeof(m_pszAESKeyBuf));
	unsigned char pszPassWD[] = "&*YKHIhfu%^d!#@";

	memcpy(m_pszAESKeyBuf, pszPassWD, sizeof(pszPassWD));
    // spdlog::info("Encryptor: {}", str(&m_pszAESKeyBuf));
    return true;
}


void Encryptor::release(){
    //spdlog::info("Encryptor: {}\n", "release");
}


bool Encryptor::encrypt(unsigned char * pszDataBuf, int nDataLen){
   // spdlog::info("Encryptor: {}\n", "encrypt");
	AES_KEY aeskey;
	AES_set_encrypt_key(m_pszAESKeyBuf, 256, &aeskey);

	int nBlockSize = 32;
	for(int i=0; i<nDataLen/nBlockSize; i++) {
		AES_encrypt(pszDataBuf+nBlockSize*i, pszDataBuf+nBlockSize*i, &aeskey);
	}
//    spdlog::info("Encryptor: {}\n", "encrypt1");

	return true;
}


bool Encryptor::decrypt(unsigned char * pszDataBuf, int nDataLen){
    //spdlog::info("Encryptor: {}\n", "decrypt");
	AES_KEY aeskey;
	AES_set_decrypt_key(m_pszAESKeyBuf, 256, &aeskey);
	int nBlockSize = 32;
	for(int i=0; i<nDataLen/nBlockSize; i++){
		AES_decrypt(pszDataBuf+nBlockSize*i, pszDataBuf+nBlockSize*i, &aeskey);
	}
    // spdlog::info("Decryptor: {}", str(&m_pszAESKeyBuf));
	return true;
}