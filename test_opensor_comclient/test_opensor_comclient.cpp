#include <opensor_com\Comm.h>

int main() {
	sor::CommClient *commClient = new sor::CommClient();
	commClient->connectToServer("127.0.0.1", "12345");
	const char *testMsg = "test message";
	commClient->sendToServer(testMsg);
	char test;
	std::cin >> test;
	return 0;
}