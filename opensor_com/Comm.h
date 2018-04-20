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

#pragma comment(lib, "Ws2_32.lib")
#pragma comment (lib, "Mswsock.lib")
#pragma comment (lib, "AdvApi32.lib")

#define DEFAULTPORT "12345"

namespace sor {
	class CommServer {
	public:
		CommServer();

		WSADATA wsaData;
		int iResult;
		const char *defaultPort = DEFAULTPORT;

		SOCKET ClientSocket;
		SOCKET ListenSocket;
		char recvbuf[512];
		int recvbuflen = 512;

		int initialize(const char* ip_addr, const char* port);
		int listenForClient();
		int receive();
	};

	class CommClient {
	public:
		CommClient();

		WSADATA wsaData;
		int iResult;
		const char *defaultPort = DEFAULTPORT;
		char recvbuf[512];
		int recvbuflen = 512;

		SOCKET ConnectSocket;

		int connectToServer(const char* ip_addr, const char* port);
		int connectToServer(const char* ip_addr);
		int sendToServer(const char *sendbuf);
	};
}