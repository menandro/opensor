#include "Comm.h"

sor::CommServer::CommServer() {
	ListenSocket = INVALID_SOCKET;
	ClientSocket = INVALID_SOCKET;
}

int sor::CommServer::initialize() {
	// Initialize Winsock
	iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
	if (iResult != 0) {
		std::cout << "WSAStartup failed: " << iResult << std::endl;
		return 1;
	}
	return 0;
}

bool sor::CommServer::isConnected() {
	if (connectionStatus == CONNECTION_STATUS_CONNECTED) {
		return true;
	}
	else
		return false;
}

int sor::CommServer::listenForClient(const char* ip_addr, const char* port) {
	server.sin_family = AF_INET;
	inet_pton(AF_INET, ip_addr, &(server.sin_addr));
	server.sin_port = htons((unsigned short)std::strtoul(port, NULL, 0));

	// Create a SOCKET for connecting to server
	ListenSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if (ListenSocket == INVALID_SOCKET) {
		std::cout << "Error creating socket: " << WSAGetLastError() << std::endl;
		WSACleanup();
		return 1;
	}

	// Setup the TCP listening socket
	iResult = bind(ListenSocket, (SOCKADDR *)&server, sizeof(server));
	if (iResult == SOCKET_ERROR) {
		std::cout << "Socket bind failed: " << WSAGetLastError() << std::endl;
		closesocket(ListenSocket);
		WSACleanup();
		return 1;
	}

	std::cout << "************** SERVER ************" << std::endl;
	std::cout << "Connection Server initialized. (" << ip_addr << ", " << port << ")" << std::endl;

	iResult = listen(ListenSocket, SOMAXCONN);
	if (iResult == SOCKET_ERROR) {
		std::cout << "Listed failed. " << WSAGetLastError();
		closesocket(ListenSocket);
		WSACleanup();
		return 1;
	}
	std::cout << "Listen socket created." << std::endl;
	connectionStatus = CONNECTION_STATUS_CONNECTING;

	// Accept a client socket
	//while (ClientSocket == INVALID_SOCKET) {
	ClientSocket = accept(ListenSocket, NULL, NULL);
	if (ClientSocket == INVALID_SOCKET) {
		std::cout << "Accept failed. " << WSAGetLastError();
		closesocket(ListenSocket);
		WSACleanup();
		return 1;
	}
	//}
	std::cout << "Client connected: (ID: " << ClientSocket << ")" << std::endl;
	connectionStatus = CONNECTION_STATUS_CONNECTED;

	// No longer need listen socket
	closesocket(ListenSocket);
	ListenSocket = INVALID_SOCKET;
	return 0;
}

int sor::CommServer::receiveFromClient(char* recvbuf, int &recvbuflen) {
	//header = {"data", "debug"}
	iResult = recv(ClientSocket, recvbuf, DEFAULTBUFFERSIZE, 0);
	if (iResult > 0) {
		recvbuflen = iResult;
		return iResult;
	}
	else if (iResult == 0) {
		std::cout << "Client closed. Shutting down server." << std::endl;
		return iResult;
	}
	else {
		std::cout << "Receive failed with error " << WSAGetLastError() << std::endl;
		closesocket(ClientSocket);
		WSACleanup();
		return iResult;
	}
}

char sor::CommServer::getChar() {
	char recvbuf[1];
	recvbuf[0] = '\0';
	iResult = recv(ClientSocket, recvbuf, 1, 0);
	if (iResult == 0) {
		std::cout << "Client closed. Shutting down server." << std::endl;
		connectionStatus = CONNECTION_STATUS_CLOSED;
	}
	else if (iResult < 0){
		std::cout << "getChar(): Receive failed with error " << WSAGetLastError() << std::endl;
		/*closesocket(ClientSocket);
		WSACleanup();*/
		connectionStatus = CONNECTION_STATUS_ERROR;
	}
	return recvbuf[0];
}

int sor::CommServer::getInt() {
	unsigned int int_data;
	char received[4];
	for (int counter = 0; counter < 4; counter++) {
		received[counter] = this->getChar();
	}
	int_data = *(unsigned int *)received;

	//std::cout << received[0] << " " << received[1] << " " << received[2] << " " << received[3] << std::endl;
	//int_data = *(unsigned int *)received;
	////iResult = recv(ClientSocket, (char*)&int_data, sizeof(int), 0);
	//if (iResult == 0) {
	//	std::cout << "Client closed. Shutting down server." << std::endl;
	//	connectionStatus = CONNECTION_STATUS_CLOSED;
	//}
	//else if (iResult < 0) {
	//	std::cout << "getInt(): Receive failed with error " << WSAGetLastError() << std::endl;
	//	/*closesocket(ClientSocket);
	//	WSACleanup();*/
	//	connectionStatus = CONNECTION_STATUS_ERROR;
	//}
	return int_data;
}

void sor::CommServer::getMessageSize(int &messageSize) {
	messageSize = this->getInt();
}

void sor::CommServer::getHeaderAndSize(std::string &header, int &messageSize) {
	//wait for first '<'
	char c = this->getChar();
	while ((c != '<') && isConnected()) {
		c = this->getChar();
	}

	c = this->getChar();
	while ((c != '>') && isConnected()) {
		//get header or ender
		header.append(std::string(1, c));
		c = this->getChar();
		if (header.length() > 10) {
			header = "error";
			return;
		}
		//std::cout << c;
	}

	//get message size
	messageSize = this->getInt();
	//std::cout << "xxx: " << messageSize << std::endl;
}

void sor::CommServer::getHeader(std::string &header) {
	//wait for first '<'
	char c = this->getChar();
	while ((c != '<') && isConnected()) {
		c = this->getChar();
	}

	c = this->getChar();
	while ((c != '>') && isConnected()) {
		//get header or ender
		header.append(std::string(1, c));
		c = this->getChar();
		if (header.length() > 20) {
			header = "error";
			return;
		}
		//std::cout << c;
	}
}

void sor::CommServer::getMessage(char* message, int messageSize) {
	if (message != NULL) {
		int receivedBytes = 0;

		while (receivedBytes < messageSize) {
			iResult = recv(ClientSocket, message + receivedBytes, messageSize - receivedBytes, 0);
			if (iResult > 0) {
				//std::cout << "res: " << iResult << std::endl;
				receivedBytes += iResult;
			}
			else if (iResult == 0) {
				std::cout << "Client closed. Shutting down server." << std::endl;
				connectionStatus = CONNECTION_STATUS_CLOSED;
				break;
			}
			else if (iResult < 0) {
				std::cout << "getMessage(): Receive failed with error " << WSAGetLastError() << std::endl;
				/*closesocket(ClientSocket);
				WSACleanup();*/
				connectionStatus = CONNECTION_STATUS_ERROR;
				break;
			}
		}
		//std::cout << "reading: " << messageSize <<  std::endl;
		//iResult = recv(ClientSocket, message, messageSize, 0);

		//if (iResult > 0) {
		//	receivedBytes += iResult;
		//	while (receivedBytes < messageSize) {
		//		//READ SOME MORE
		//		iResult = recv(ClientSocket, message, messageSize, 0);
		//	}
		//}
		//std::cout << message << std::endl;

		//if (iResult == 0) {
		//	std::cout << "Client closed. Shutting down server." << std::endl;
		//	connectionStatus = CONNECTION_STATUS_CLOSED;
		//}
		//else if (iResult < 0) {
		//	std::cout << "getMessage(): Receive failed with error " << WSAGetLastError() << std::endl;
		//	/*closesocket(ClientSocket);
		//	WSACleanup();*/
		//	connectionStatus = CONNECTION_STATUS_ERROR;
		//}
		message[messageSize] = '\0'; //insert null at the end for printing purposes
		/*std::cout << iResult << std::endl;*/
	}
}

int sor::CommServer::sendToClient(char *sendbuf, int sendbuflen) {
	iResult = send(ClientSocket, sendbuf, sendbuflen, 0);
	if (iResult == SOCKET_ERROR) {
		std::cout << "Send failed with error: " << WSAGetLastError() << std::endl;
		closesocket(ClientSocket);
		WSACleanup();
		return 1;
	}
	return 0;
}

int sor::CommServer::closeClientConnection() {
	// shutdown the connection since no more data will be sent
	iResult = shutdown(ClientSocket, SD_SEND);
	if (iResult == SOCKET_ERROR) {
		std::cout << "Shutdown failed with error: " << WSAGetLastError() << std::endl;
		closesocket(ClientSocket);
		WSACleanup();
		return 1;
	}
	closesocket(ClientSocket);
	ClientSocket = INVALID_SOCKET;
	std::cout << "Client terminated." << std::endl;
	return 0;
}

int sor::CommServer::closeConnection() {
	// cleanup
	closesocket(ClientSocket);
	WSACleanup();
	std::cout << "Server closed." << std::endl;
	ClientSocket = INVALID_SOCKET;
	return 0;
}



//*************** TRASH BIN *******************//
//*************** TRASH BIN *******************//
//*************** TRASH BIN *******************//
int sor::CommServer::receive() {
	//// Receive until the peer shuts down the connection
	//int iSendResult;
	//do {

	//	iResult = recv(ClientSocket, recvbuf, recvbuflen, 0);
	//	if (iResult > 0) {
	//		printf("Bytes received: %d\n", iResult);

	//		// Echo the buffer back to the sender
	//		iSendResult = send(ClientSocket, recvbuf, iResult, 0);
	//		if (iSendResult == SOCKET_ERROR) {
	//			printf("send failed with error: %d\n", WSAGetLastError());
	//			closesocket(ClientSocket);
	//			WSACleanup();
	//			return 1;
	//		}
	//		printf("Bytes sent: %d\n", iSendResult);
	//	}
	//	else if (iResult == 0)
	//		printf("Connection closing...\n");
	//	else {
	//		printf("recv failed with error: %d\n", WSAGetLastError());
	//		closesocket(ClientSocket);
	//		WSACleanup();
	//		return 1;
	//	}

	//} while (iResult > 0);

	//// shutdown the connection since we're done
	//iResult = shutdown(ClientSocket, SD_SEND);
	//if (iResult == SOCKET_ERROR) {
	//	printf("shutdown failed with error: %d\n", WSAGetLastError());
	//	closesocket(ClientSocket);
	//	WSACleanup();
	//	return 1;
	//}

	//// cleanup
	//closesocket(ClientSocket);
	//WSACleanup();

	return 0;
}

void sor::CommServer::getDebugMessage(int messageSize) {
	iResult = recv(ClientSocket, debugMessage, messageSize, 0);
	if (iResult == 0) {
		std::cout << "Client closed. Shutting down server." << std::endl;
		connectionStatus = CONNECTION_STATUS_CLOSED;
	}
	else if (iResult < 0) {
		std::cout << "Receive failed with error " << WSAGetLastError() << std::endl;
		/*closesocket(ClientSocket);
		WSACleanup();*/
		connectionStatus = CONNECTION_STATUS_ERROR;
	}
	debugMessage[messageSize] = '\0'; //insert null at the end for printing purposes
}

int sor::CommServer::receiveOneXmlFromClient() {
	//wait for first '<'
	std::string header = "";
	int messageSize;

	char c = this->getChar();
	while ((c != '<') && isConnected()) {
		c = this->getChar();
	}

	c = this->getChar();
	while ((c != '>') && isConnected()) {
		//get header or ender
		header.append(std::string(1, c));
		c = this->getChar();
	}
	//std::cout << header << std::endl;

	if (header.compare("dbg") == 0) {
		//read message size
		messageSize = this->getInt();
		//printf("%d\n", messageSize);

		//read message
		getDebugMessage(messageSize);
		//std::cout << debugMessage << std::endl;
	}

	else if ((header.compare("data") == 0)) {

	}
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