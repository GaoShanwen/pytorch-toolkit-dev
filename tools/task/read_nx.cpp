////////////////////////////////////////////////////////////////////////////////////
// author: gaowenjie
// email: gaowenjie@rongxwy.com
// date: 2024.07.05
// filenaem: read_nx.cpp
// function: load binary file check the file format and read the data.
// note: g++ ./project/debug/read_nx.cpp -o ./project/debug/read_nx -lssl -lcrypto
////////////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <sstream>
#include <openssl/aes.h>
#include <openssl/err.h>
#include <iomanip>


void LOGE(const std::string& message1, const std::string& message2) {
    std::cerr << message1 << message2 << std::endl;
}


unsigned char * read_binary_file(const std::string filename, int &nDataLen) {
    std::ifstream fin;
    fin.open(filename.c_str(), std::ios::binary);
    if (!fin.is_open()) {
        LOGE("buffer2index, 打开索引文件失败！", filename.c_str());
        // return false;
    }

    fin.seekg(0, std::ios::end);
    nDataLen = fin.tellg();
    unsigned char *pszDataBuf = new(std::nothrow) unsigned char[nDataLen];
    // pszDataBuf = unsigned char[nDataLen];
    if (pszDataBuf == NULL) {
        LOGE("decrypt_load", "pszDataBuf 分配内存失败！！");
        // return false;
    }
    fin.seekg(0, std::ios::beg);
    fin.read((char *) pszDataBuf, nDataLen);
    fin.close();
    std::cout << "read_key_file, 读取索引文件成功！" << std::endl;

    std::cout << "end:" << nDataLen << std::endl;
    return pszDataBuf; // true;
}


unsigned char * AES_ECB_Decrypt(unsigned char * pszDataBuf, int nDataLen) {
	AES_KEY aeskey;
    unsigned char m_pszAESKeyBuf[32];
	memset(m_pszAESKeyBuf, 0, sizeof(m_pszAESKeyBuf));
	unsigned char pszPassWD[] = "&*YKHIhfu%^d!#@";

	memcpy(m_pszAESKeyBuf, pszPassWD, sizeof(pszPassWD));

	AES_set_decrypt_key(m_pszAESKeyBuf, 256, &aeskey);
	int nBlockSize = 32;
	for(int i=0; i<nDataLen/nBlockSize; i++) {
		AES_decrypt(pszDataBuf+nBlockSize*i, pszDataBuf+nBlockSize*i, &aeskey);
	}
    // LOGE("Decryptor: ", m_pszAESKeyBuf.toStdString());
    
	return pszDataBuf;
}


using namespace std;
bool buffer2index(unsigned char * pszDataBuf, int nDataLen) {
    int m_nListLen, m_nFeaLen, m_nMaxClassID;
    cout << "int size: " << sizeof(int) << endl;
    int nOffset = 0;
    memcpy((unsigned char *) &m_nListLen, pszDataBuf + nOffset, sizeof(int));
    nOffset += sizeof(int);
    memcpy((unsigned char *) &m_nFeaLen, pszDataBuf + nOffset, sizeof(int));
    nOffset += sizeof(int);
    memcpy((unsigned char *) &m_nMaxClassID, pszDataBuf + nOffset, sizeof(int));
    nOffset += sizeof(int);

    if (m_nListLen != 120000 && m_nFeaLen != 128 && m_nMaxClassID != 120000) {
        LOGE("m_nListLen: ", to_string(m_nListLen));
        LOGE("m_nFeaLen: ", to_string(m_nFeaLen));
        LOGE("m_nMaxClassID: ", to_string(m_nMaxClassID));
        return false;
    }
    else {
        cout << "m_nListLen: " << m_nListLen << ", m_nFeaLen: " << m_nFeaLen << ", m_nMaxClassID: " << m_nMaxClassID << endl;
    }

    int m_nInsertNum, m_nMaxLoadTimes, m_nMaxSearchTimes, m_nMaxRegistTimes, m_nLoadTimes, m_nSearchTimes, m_nRegistTimes;

    memcpy((unsigned char *) &m_nInsertNum, pszDataBuf + nOffset, sizeof(int));
    nOffset += sizeof(int);
    memcpy((unsigned char *) &m_nMaxLoadTimes, pszDataBuf + nOffset, sizeof(int));
    nOffset += sizeof(int);
    memcpy((unsigned char *) &m_nMaxSearchTimes, pszDataBuf + nOffset, sizeof(int));
    nOffset += sizeof(int);
    memcpy((unsigned char *) &m_nMaxRegistTimes, pszDataBuf + nOffset, sizeof(int));
    nOffset += sizeof(int);
    memcpy((unsigned char *) &m_nLoadTimes, pszDataBuf + nOffset, sizeof(int));
    nOffset += sizeof(int);
    memcpy((unsigned char *) &m_nSearchTimes, pszDataBuf + nOffset, sizeof(int));
    nOffset += sizeof(int);
    memcpy((unsigned char *) &m_nRegistTimes, pszDataBuf + nOffset, sizeof(int));
    nOffset += sizeof(int);

    cout << "m_nInsertNum: " << m_nInsertNum << ", m_nMaxLoadTimes: " << m_nMaxLoadTimes << ", m_nMaxSearchTimes: " << m_nMaxSearchTimes << endl;
    
    float *m_pfFeaMat;
    m_pfFeaMat = new(std::nothrow) float[m_nFeaLen * m_nListLen];
    memcpy((unsigned char *) m_pfFeaMat, pszDataBuf + nOffset, sizeof(float) * m_nFeaLen * m_nListLen);
    nOffset += sizeof(float) * m_nFeaLen * m_nListLen;
    // LOGE("buffer2index，memcpy2  nOffset=%d", to_string(nOffset));
    cout << "m_pfFeaMat: " << m_pfFeaMat[0] << endl;
    int *m_pnClsIDMap;
    m_pnClsIDMap = new(std::nothrow) int[m_nListLen];
    memcpy((unsigned char *) m_pnClsIDMap, pszDataBuf + nOffset, sizeof(int) * m_nListLen);
    nOffset += sizeof(int) * m_nListLen;
    // LOGE("buffer2index，memcpy3  nOffset=%d", to_string(nOffset));
    cout << "m_pnClsIDMap[    0]:" << m_pnClsIDMap[0] << endl;
    cout << "m_pnClsIDMap[  200]:" << m_pnClsIDMap[200] << endl;
    cout << "m_pnClsIDMap[ 1000]:" << m_pnClsIDMap[1000] << endl;
    cout << "m_pnClsIDMap[15000]:" << m_pnClsIDMap[15000] << endl;
    cout << "m_pnClsIDMap[30000]:" << m_pnClsIDMap[30000] << endl;
    cout << "m_pnClsIDMap[60000]:" << m_pnClsIDMap[60000] << endl;

    int *m_pnClsIDMapSearchTimes;
    m_pnClsIDMapSearchTimes = new(std::nothrow) int[m_nListLen];
    int *m_pnClsIDMapSelectTimes;
    m_pnClsIDMapSelectTimes = new(std::nothrow) int[m_nListLen];
    int *m_pnClsIDMapSameTimes;
    m_pnClsIDMapSameTimes = new(std::nothrow) int[m_nListLen];
    int *m_pnClsIDMapReplaceTimes;
    m_pnClsIDMapReplaceTimes = new(std::nothrow) int[m_nListLen];
    memcpy((unsigned char *) m_pnClsIDMapSearchTimes, pszDataBuf + nOffset, sizeof(int) * m_nListLen);
    nOffset += sizeof(int) * m_nListLen;
    // LOGE("buffer2index, memcpy4  nOffset=%d", to_string(nOffset));

    memcpy((unsigned char *) m_pnClsIDMapSelectTimes, pszDataBuf + nOffset, sizeof(int) * m_nListLen);
    nOffset += sizeof(int) * m_nListLen;
    // LOGE("buffer2index，memcpy5  nOffset=%d", to_string(nOffset));

    memcpy((unsigned char *) m_pnClsIDMapSameTimes, pszDataBuf + nOffset, sizeof(int) * m_nListLen);
//    nOffset += sizeof(long int) * m_nListLen;
    nOffset += sizeof(int) * m_nListLen;
    // LOGE("buffer2index，memcpy6  nOffset=%d", to_string(nOffset));

    memcpy((unsigned char *) m_pnClsIDMapReplaceTimes, pszDataBuf + nOffset, sizeof(int) * m_nListLen);
//    nOffset += sizeof(long int) * m_nListLen;
    nOffset += sizeof(int) * m_nListLen;
    // LOGE("buffer2index，memcpy7  nOffset=%d", to_string(nOffset));

    int m_OssLength = 150;
    int sSize = sizeof(char) * m_OssLength * m_nListLen;
    char * m_pnClsIDOss = new(std::nothrow) char[m_OssLength * m_nListLen];
    // LOGE("buffer2index，m_OssLength=%d size=%d", m_OssLength, sSize);
    memcpy((unsigned char *) m_pnClsIDOss, pszDataBuf + nOffset, sizeof(char) * m_OssLength * m_nListLen);
    nOffset += sizeof(char) * m_OssLength * m_nListLen;
    // LOGE("buffer2index，memcpy8  nOffset=%d", to_string(nOffset));

    int nums[6] = {0, 200, 1000, 15000, 30000, 60000};
    string m_pnClsID0 = "";
    for (int j = 0; j < 6; j++) {
        for (int i=0; i<150; i++) {
            m_pnClsID0 += m_pnClsIDOss[i+nums[j]*150];
        }
        cout << "m_pnClsIDOss[" << setw(6) << setfill('0') << nums[j] << "]: " << m_pnClsID0 << endl;
        m_pnClsID0 = "";
        // LOGE("buffer2index，m_pnClsIDOss[]: ", m_pnClsID0.c_str());
    }

    delete[] m_pfFeaMat;
    delete[] m_pnClsIDMap;
    delete[] m_pnClsIDMapSearchTimes;
    delete[] m_pnClsIDMapSelectTimes;
    delete[] m_pnClsIDMapSameTimes;
    delete[] m_pnClsIDMapReplaceTimes;
    delete[] m_pnClsIDOss;

    return true;
}

int main(int argc, char *argv[]) {
    const char *infile = argv[1];
    // std::string infile = "dataset/test/modelnew1.nx";
    int data_length;
    unsigned char* out = read_binary_file(infile, data_length);
    
    // std::cout << "end:" << data_length << std::endl;
    out = AES_ECB_Decrypt(out, data_length);  // 读取原始 AES 密文到 outfile
    std::cout << "end:" << data_length << std::endl;
    bool flag = buffer2index(out, data_length);  // 解析 NX 文件内容
 
    return 0;
}
