#include <opensor_com\Comm.h>

int main() {
	sor::CommServer *commServer = new sor::CommServer();
	commServer->initialize("127.0.0.1", "12345");
	commServer->listenForClient();
	commServer->receive();
	char test;
	std::cin >> test;
	return 0;
}