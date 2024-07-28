#ifndef __ENCRYPTOR_H__
#define __ENCRYPTOR_H__

class Encryptor{
	public:
		Encryptor();
		~Encryptor();
		bool init();
		void release();
        bool encrypt(unsigned char * pszDataBuf, int nDataLen);
        bool decrypt(unsigned char * pszDataBuf, int nDataLen);

	private:
		unsigned char m_pszAESKeyBuf[32];
};

#endif // __ENCRYPTOR_H__