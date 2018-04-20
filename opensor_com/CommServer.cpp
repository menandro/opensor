#include "Comm.h"

sor::CommServer::CommServer() {
	ListenSocket = INVALID_SOCKET;
	ClientSocket = INVALID_SOCKET;
}

int sor::CommServer::initialize(const char* ip_addr, const char* port) {
	// Initialize Winsock
	iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
	if (iResult != 0) {
		printf("WSAStartup failed: %d\n", iResult);
		return 1;
	}

	sockaddr_in server;
	server.sin_family = AF_INET;
	inet_pton(AF_INET, ip_addr, &(server.sin_addr));
	//service.sin_addr.s_addr = inet_addr("127.0.0.1");
	server.sin_port = (unsigned short)std::strtoul(port, NULL, 0);

	// Create a SOCKET for connecting to server
	ListenSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if (ListenSocket == INVALID_SOCKET) {
		printf("Error at socket(): %ld\n", WSAGetLastError());
		WSACleanup();
		return 1;
	}
	else {
		std::cout << "Socket created: " << ListenSocket << std::endl;
	}

	// Setup the TCP listening socket
	iResult = bind(ListenSocket, (SOCKADDR *)&server, sizeof(server));
	if (iResult == SOCKET_ERROR) {
		printf("bind failed with error: %d\n", WSAGetLastError());
		//freeaddrinfo(result);
		closesocket(ListenSocket);
		WSACleanup();
		return 1;
	}
	else {
		std::cout << "Socket bound." << std::endl;
	}

	//freeaddrinfo(result);
	return 0;
}


int sor::CommServer::listenForClient() {
	iResult = listen(ListenSocket, SOMAXCONN);
	if (iResult == SOCKET_ERROR) {
		printf("listen failed with error: %d\n", WSAGetLastError());
		closesocket(ListenSocket);
		WSACleanup();
		return 1;
	}
	else {
		std::cout << "Listen socket created." << std::endl;
	}

	// Accept a client socket
	while (ClientSocket == INVALID_SOCKET) {
		ClientSocket = accept(ListenSocket, NULL, NULL);
		std::cout << ClientSocket << std::endl;
		/*if (ClientSocket == INVALID_SOCKET) {
			printf("accept failed with error: %d\n", WSAGetLastError());
			closesocket(ListenSocket);
			WSACleanup();
			return 1;
		}*/
	}
	std::cout << "Client Accepted: " << ClientSocket << std::endl;
	// No longer need server socket
	closesocket(ListenSocket);

	return 0;
}

int sor::CommServer::receive() {
	// Receive until the peer shuts down the connection
	int iSendResult;
	do {

		iResult = recv(ClientSocket, recvbuf, recvbuflen, 0);
		if (iResult > 0) {
			printf("Bytes received: %d\n", iResult);

			// Echo the buffer back to the sender
			iSendResult = send(ClientSocket, recvbuf, iResult, 0);
			if (iSendResult == SOCKET_ERROR) {
				printf("send failed with error: %d\n", WSAGetLastError());
				closesocket(ClientSocket);
				WSACleanup();
				return 1;
			}
			printf("Bytes sent: %d\n", iSendResult);
		}
		else if (iResult == 0)
			printf("Connection closing...\n");
		else {
			printf("recv failed with error: %d\n", WSAGetLastError());
			closesocket(ClientSocket);
			WSACleanup();
			return 1;
		}

	} while (iResult > 0);

	// shutdown the connection since we're done
	iResult = shutdown(ClientSocket, SD_SEND);
	if (iResult == SOCKET_ERROR) {
		printf("shutdown failed with error: %d\n", WSAGetLastError());
		closesocket(ClientSocket);
		WSACleanup();
		return 1;
	}

	// cleanup
	closesocket(ClientSocket);
	WSACleanup();

	return 0;
}




//int sor::CommServer::initialize() {
//	struct addrinfo serverInfo;
//	//struct addrinfo result = NULL;
//	//struct addrinfo hints;
//	
//	// Initialize Winsock
//	iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
//	if (iResult != 0) {
//		printf("WSAStartup failed: %d\n", iResult);
//		return 1;
//	}
//
//	/*ZeroMemory(&hints, sizeof(hints));
//	hints.ai_family = AF_INET;
//	hints.ai_socktype = SOCK_STREAM;
//	hints.ai_protocol = IPPROTO_TCP;
//	hints.ai_flags = AI_PASSIVE;*/
//
//	// Resolve the server address and port
//	/*iResult = getaddrinfo("127.0.0.1", "12345", &hints, &result);
//	if (iResult != 0) {
//		printf("getaddrinfo failed with error: %d\n", iResult);
//		WSACleanup();
//		return 1;
//	}*/
//	
//	ZeroMemory(&serverInfo, sizeof(serverInfo));
//	serverInfo.ai_family = AF_INET;
//	serverInfo.ai_socktype = SOCK_STREAM;
//	serverInfo.ai_protocol = IPPROTO_TCP;
//	inet_pton(AF_INET, "127.0.0.1", &(serverInfo.ai_addr));
//	serverInfo.ai_flags = AI_PASSIVE;
//	serverInfo.ai_addrlen = INET_ADDRSTRLEN;
//
//	sockaddr_in service;
//	service.sin_family = AF_INET;
//	inet_pton(AF_INET, "127.0.0.1", &(service.sin_addr));
//	//service.sin_addr.s_addr = inet_addr("127.0.0.1");
//	service.sin_port = htons(12345);
//
//	char str[INET_ADDRSTRLEN];
//	inet_ntop(AF_INET, &(serverInfo.ai_addr), str, INET_ADDRSTRLEN);
//	std::cout << "Server: " << str << " Pointer: " << serverInfo.ai_addr << std::endl;
//	std::cout << "Address length: " << serverInfo.ai_addrlen << std::endl;
//
//	// Create a SOCKET for connecting to server
//	ListenSocket = socket(serverInfo.ai_family, serverInfo.ai_socktype, serverInfo.ai_protocol);
//	if (ListenSocket == INVALID_SOCKET) {
//		printf("Error at socket(): %ld\n", WSAGetLastError());
//		WSACleanup();
//		return 1;
//	}
//	else {
//		std::cout << "Socket created: " << ListenSocket << std::endl;
//	}
//
//	// Setup the TCP listening socket
//	iResult = bind(ListenSocket, (SOCKADDR *)&serverInfo.ai_addr, (int)serverInfo.ai_addrlen);
//	//iResult = bind(ListenSocket, (SOCKADDR *)&service, sizeof(service));
//	if (iResult == SOCKET_ERROR) {
//		printf("bind failed with error: %d\n", WSAGetLastError());
//		//freeaddrinfo(result);
//		closesocket(ListenSocket);
//		WSACleanup();
//		return 1;
//	}
//	else {
//		std::cout << "Socket bound." << std::endl;
//	}
//
//	//freeaddrinfo(result);
//	return 0;
//}