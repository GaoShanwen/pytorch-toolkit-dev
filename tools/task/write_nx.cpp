////////////////////////////////////////////////////////////////////////////////////
// author: gaowenjie
// email: gaowenjie@rongxwy.com
// date: 2024.07.05
// filenaem: write_nx.cpp
// function: convert binary file to encrypted binary file.
// use case: g++ ./tools/task/write_nx.cpp -o ./tools/task/write_nx.so -lssl -lcrypto
////////////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <sstream>
#include <openssl/aes.h>
#include <openssl/err.h>


void LOGE(const std::string& message1, const std::string& message2) {
    std::cerr << message1 << message2 << std::endl;
}


unsigned char * read_binary_file(const std::string filename, int &nDataLen) {
    std::ifstream fin;
    fin.open(filename.c_str(), std::ios::binary);
    if (!fin.is_open()) {
        LOGE("buffer2index, 打开索引文件失败！", filename.c_str());
    }

    fin.seekg(0, std::ios::end);
    nDataLen = fin.tellg();
    unsigned char *pszDataBuf = new(std::nothrow) unsigned char[nDataLen];
    if (pszDataBuf == NULL) {
        LOGE("decrypt_load", "pszDataBuf 分配内存失败！！");
    }
    fin.seekg(0, std::ios::beg);
    fin.read((char *) pszDataBuf, nDataLen);
    fin.close();
    std::cout << "read_key_file, 读取索引文件成功！" << std::endl;

    return pszDataBuf; // true;
}


void write_binary_file(const std::string strFileName, unsigned char *pszDataBuf, int nDataLen) {
    std::ofstream fout;
    fout.open(strFileName.c_str(), std::ios::binary);
    fout.write((char *) pszDataBuf, nDataLen);
    fout.close();
}


unsigned char * AES_ECB_encrypt(unsigned char * pszDataBuf, int nDataLen) {
	AES_KEY aeskey;
    unsigned char m_pszAESKeyBuf[32];
	memset(m_pszAESKeyBuf, 0, sizeof(m_pszAESKeyBuf));
	unsigned char pszPassWD[] = "&*YKHIhfu%^d!#@";

	memcpy(m_pszAESKeyBuf, pszPassWD, sizeof(pszPassWD));

	AES_set_encrypt_key(m_pszAESKeyBuf, 256, &aeskey);
	int nBlockSize = 32;
	for(int i=0; i<nDataLen/nBlockSize; i++) {
		AES_encrypt(pszDataBuf+nBlockSize*i, pszDataBuf+nBlockSize*i, &aeskey);
	}
    
	return pszDataBuf;
}


int main(int argc, char *argv[]) {
    const char *infile = argv[1];
    const char *outfile = argv[2];

    int data_length;
    unsigned char* buffer = read_binary_file(infile, data_length);  // 读取原始 NX 文件到内存
    
    // std::cout << "end:" << data_length << std::endl;
    buffer = AES_ECB_encrypt(buffer, data_length);  // 读取原始 AES 密文到 outfile
    std::cout << "end:" << data_length << std::endl;
    write_binary_file(outfile, buffer, data_length);  // 写入解密后文件
    std::cout << "end:" << "写入成功！" << std::endl;
    return 0;
}
