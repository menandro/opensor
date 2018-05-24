#include "Comm.h"

sor::CommClient::CommClient() {
	ConnectSocket = INVALID_SOCKET;
}

int sor::CommClient::connectToServer(const char* ip_addr) {
	const char * port = defaultPort;
	return connectToServer(ip_addr, port);
}

int sor::CommClient::connectToServer(std::string ip_addr, std::string port) {
//int sor::CommClient::connectToServer(const char* ip_addr, const char* port) {
	// Initialize Winsock
	iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
	if (iResult != 0) {
		std::cout << "WSAStartup failed: " << iResult << std::endl;
		return 1;
	}
	
	struct sockaddr_in client;
	client.sin_family = AF_INET;
	inet_pton(AF_INET, ip_addr.c_str(), &(client.sin_addr));
	client.sin_port = htons((unsigned short)std::strtoul(port.c_str(), NULL, 0));

	ConnectSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if (ConnectSocket == INVALID_SOCKET) {
		std::cout << "Error creating socket: " << WSAGetLastError() << std::endl;
		WSACleanup();
		return 1;
	}

	// Connect to server.
	iResult = connect(ConnectSocket, (SOCKADDR*)&client, sizeof(client));
	if (iResult == SOCKET_ERROR) {
		std::cout << "Failed to connect to server: " << WSAGetLastError() << std::endl;
		closesocket(ConnectSocket);
		ConnectSocket = INVALID_SOCKET;
	}
	std::cout << "Connected to server. " << std::endl;

	if (ConnectSocket == INVALID_SOCKET) {
		std::cout << "Unable to connect to server. Server not found." << std::endl;
		WSACleanup();
		return 1;
	}
}


int sor::CommClient::sendToServer(char *sendbuf, int sendbuflen) {
	iResult = send(ConnectSocket, sendbuf, sendbuflen, 0);
	if (iResult == SOCKET_ERROR) {
		std::cout << "Send failed: " << WSAGetLastError() << std::endl;
		closesocket(ConnectSocket);
		WSACleanup();
		return 1;
	}
	return 0;
}

int sor::CommClient::receiveFromServer(char *recvbuf, int &recvbuflen) {
	iResult = recv(ConnectSocket, recvbuf, DEFAULTBUFFERSIZE, 0);
	if (iResult > 0) {
		recvbuflen = iResult;
	}
	else if (iResult == 0)
		std::cout << "Connection closing... " << std::endl;
	else {
		std::cout << "Receive failed with error " << WSAGetLastError() << std::endl;
		closesocket(ConnectSocket);
		WSACleanup();
		return 1;
	}
	return 0;
}

int sor::CommClient::closeConnection() {
	// shutdown the connection since no more data will be sent
	iResult = shutdown(ConnectSocket, SD_SEND);
	if (iResult == SOCKET_ERROR) {
		std::cout << "Shutdown failed with error: " << WSAGetLastError() << std::endl;
		closesocket(ConnectSocket);
		WSACleanup();
		return 1;
	}

	// cleanup
	closesocket(ConnectSocket);
	WSACleanup();
	ConnectSocket = INVALID_SOCKET;
	std::cout << "Client closed." << std::endl;
	return 0;
}


//int sor::CommClient::connectToServer(const char* ip_addr, const char* port) {
//	// Initialize Winsock
//	iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
//	if (iResult != 0) {
//		printf("WSAStartup failed with error: %d\n", iResult);
//		return 1;
//	}
//
//	/*struct addrinfo *result = NULL;
//	struct addrinfo	*ptr = NULL;
//	struct addrinfo	hints;*/
//	struct addrinfo serverInfo;
//
//	/*ZeroMemory(&hints, sizeof(hints));
//	hints.ai_family = AF_INET;
//	hints.ai_socktype = SOCK_STREAM;
//	hints.ai_protocol = IPPROTO_TCP;*/
//
//	// Resolve the server address and port
//	/*iResult = getaddrinfo(ip_addr, port, &hints, &result);
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
//	char str[INET_ADDRSTRLEN];
//	inet_ntop(AF_INET, &(serverInfo.ai_addr), str, INET_ADDRSTRLEN);
//	std::cout << "Server: " << str << std::endl;
//
//	ConnectSocket = socket(serverInfo.ai_family, serverInfo.ai_socktype, serverInfo.ai_protocol);
//	if (ConnectSocket == INVALID_SOCKET) {
//		printf("socket failed with error: %ld\n", WSAGetLastError());
//		WSACleanup();
//		return 1;
//	}
//	std::cout << "Socket opened socket: " << ConnectSocket << std::endl;
//	// Connect to server.
//	iResult = connect(ConnectSocket, serverInfo.ai_addr, (int)serverInfo.ai_addrlen);
//	if (iResult == SOCKET_ERROR) {
//		printf("FAILED TO CONNECT: %ld\n", WSAGetLastError());
//		closesocket(ConnectSocket);
//		ConnectSocket = INVALID_SOCKET;
//	}
//	std::cout << ConnectSocket << " " << INVALID_SOCKET << std::endl;
//	//// Attempt to connect to an address until one succeeds
//	//for (ptr = result; ptr != NULL;ptr = ptr->ai_next) {
//
//	//	// Create a SOCKET for connecting to server
//	//	ConnectSocket = socket(ptr->ai_family, ptr->ai_socktype,
//	//		ptr->ai_protocol);
//	//	if (ConnectSocket == INVALID_SOCKET) {
//	//		printf("socket failed with error: %ld\n", WSAGetLastError());
//	//		WSACleanup();
//	//		return 1;
//	//	}
//
//	//	// Connect to server.
//	//	iResult = connect(ConnectSocket, ptr->ai_addr, (int)ptr->ai_addrlen);
//	//	if (iResult == SOCKET_ERROR) {
//	//		closesocket(ConnectSocket);
//	//		ConnectSocket = INVALID_SOCKET;
//	//		continue;
//	//	}
//	//	break;
//	//}
//
//	//freeaddrinfo(result);
//
//	if (ConnectSocket == INVALID_SOCKET) {
//		printf("Unable to connect to server!\n");
//		WSACleanup();
//		return 1;
//	}
//}