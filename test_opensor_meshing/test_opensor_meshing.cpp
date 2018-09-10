#include <opensor_meshing/Meshing.h>
#include <opensor_velodyne/Velodyne.h>
#include <stdio.h>
#include <iostream>

int test_receiving() {
	Velodyne *velodyne = new Velodyne("192.168.1.102", 13000, CaptureMode::TCP);
	if (!velodyne->isOpen()) {
		std::cout << "Can't open velodyne." << std::endl;
		return 0;
	}

	return 0;
}

int main()
{
	test_receiving();
    return 0;
}

