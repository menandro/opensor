#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#include <stdio.h>
#include <iostream>
#include <ostream>
#include <string>

#pragma comment(lib, "Ws2_32.lib")
#pragma comment (lib, "Mswsock.lib")
#pragma comment (lib, "AdvApi32.lib")

#define DEFAULTPORT "12345"
#define DEFAULTBUFFERSIZE 512

namespace sor {
	class CommServer {
	public:
		CommServer();

		enum ConnectionStatus {
			CONNECTION_STATUS_CONNECTED,
			CONNECTION_STATUS_CLOSED,
			CONNECTION_STATUS_ERROR,
			CONNECTION_STATUS_CONNECTING
		} connectionStatus = CONNECTION_STATUS_CLOSED;

		bool isConnected();

		WSADATA wsaData;
		int iResult;
		const char *defaultPort = DEFAULTPORT;
		sockaddr_in server;
		const char* ip_addr;
		const char* port;

		SOCKET ClientSocket;
		SOCKET ListenSocket;
		//char recvbuf[DEFAULTBUFFERSIZE];

		int initialize();
		int listenForClient(const char* ip_addr, const char* port);
		int closeConnection();
		int closeClientConnection();
		int receiveFromClient(char* recvbuf, int &recvbuflen); //more general

		char getChar();
		int getInt();

		//general message (debug or image) format of sending: "<dbg>INT4messagemessagemessage..." INT4 is a 4-byte char array
		void getHeaderAndSize(std::string &header, int &messageSize);
		void getHeader(std::string &header);
		void getMessage(char* message, int messageSize);
		void getMessageSize(int &messageSize);

		//trash??
		char dataMessage[100]; //change this to size of image
		int receiveOneXmlFromClient();
		//debug messages (for hololens) //important to separate message from images
		char debugMessage[160];
		void getDebugMessage(int messageSize);
		int receive();
		int sendToClient(char* sendbuf, int sendbuflen);
		

	};

	class CommClient {
	public:
		CommClient();

		WSADATA wsaData;
		int iResult;
		const char *defaultPort = DEFAULTPORT;

		SOCKET ConnectSocket;

		int connectToServer(std::string ip_addr, std::string port);
		//int connectToServer(const char* ip_addr, const char* port);
		int connectToServer(const char* ip_addr);
		int receiveFromServer(char* recvbuf, int &recvbuflen);
		int sendToServer(char *sendbuf, int sendbuflen);
		int closeConnection();
	};
}