#include "Velodyne.h"
#include <iostream>

Velodyne::Velodyne(boost::asio::ip::address& address, unsigned short port = 2368)
{
	this->open(address, port);
}

Velodyne::Velodyne(const char *address_str, unsigned short port = 2368)
{
	boost::asio::ip::address address = boost::asio::ip::address::from_string(address_str);
	this->open(address, port);
}

Velodyne::~Velodyne() {
	close();
	//delete packet;
}

void Velodyne::calibFileRead(const char *filename)
{
	using boost::property_tree::ptree;
	ptree pt;
	//std::string fname = std::string(filename, 11);
	if (!boost::filesystem::exists(filename)) {
		std::cout << "Calibration file not found. Using default corrections. " << std::endl;
		isCalibrated = false;
		return;
	}

	isCalibrated = true;
	boost::property_tree::xml_parser::read_xml(filename, pt);
	calibParams.empty();
	std::cout << "Reading Calibration File...";
	//std::cout << pt.front << std::endl;
	BOOST_FOREACH(boost::property_tree::ptree::value_type const& v_db, pt.get_child("boost_serialization").get_child("DB")) {
		if (v_db.first == "points_") {
			laserCount = v_db.second.get<int>("count");

			ptree pt_points = (ptree)v_db.second;
			BOOST_FOREACH(ptree::value_type &v_points, pt_points) {
				//std::cout << v_points.first << std::endl;
				if (v_points.first == "item") {
					ptree pt_item = (ptree)v_points.second;
					BOOST_FOREACH(ptree::value_type &v_item, pt_item) {
						//std::cout << v_item.first << std::endl;
						if (v_item.first == "px") {
							CalibParams cp;
							cp.vertCorrection = v_item.second.get<double>("vertCorrection_");
							cp.distCorrection = v_item.second.get<double>("distCorrection_");
							cp.distCorrectionX = v_item.second.get<double>("distCorrectionX_");
							cp.distCorrectionY = v_item.second.get<double>("distCorrectionY_");
							cp.focalDistance = v_item.second.get<double>("focalDistance_");
							cp.focalSlope = v_item.second.get<double>("focalSlope_");
							cp.horizOffsetCorrection = v_item.second.get<double>("horizOffsetCorrection_");
							cp.rotCorrection = v_item.second.get<double>("rotCorrection_");
							cp.laserId = v_item.second.get<unsigned char>("id_");
							this->calibParams.push_back(cp);
							//std::cout << cp.vertCorrection << std::endl;
						}
					}
				}
			}
		}
	}
	std::cout << "DONE." << std::endl;
}

bool Velodyne::open(boost::asio::ip::address& address, unsigned short port = 2368)
{
	this->calibFileRead("default.xml");
	//this->vertAngle = std::vector<double>(64);
	/*for (int k = 0; k < 64; k++) {
		double vertAng = 2.0 - (double)k * (26.8 / 63.0);
		this->vertAngle.push_back(vertAng);
	}*/

	if (isRunning()) {
		std::cout << "Velodyne capture already running." << std::endl;
		//close();
	}

	this->address = address;
	this->port = port;

	try {
		this->socket = new boost::asio::ip::udp::socket(ioService, boost::asio::ip::udp::endpoint(this->address, this->port));
	}
	catch (const std::exception& e) {
		std::cerr << "Failed to open socket: [" << e.what() << "]" << std::endl;
		std::cerr << "Trying different address..." << std::endl;
		delete socket;

		try {
			socket = new boost::asio::ip::udp::socket(ioService, 
				boost::asio::ip::udp::endpoint(boost::asio::ip::address_v4::any(), this->port));
			std::cout << "Opened socket from broadcast in port: " << this->port << std::endl;
		}
		catch (const std::exception& e) {
			std::cerr << "Failed to open socket:: [" << e.what() << "]" << std::endl;
			return false;
		}
	}

	try {
		ioService.run();
	}
	catch (const std::exception& e) {
		std::cerr << "Failed to start io service: [" << e.what() << "]" << std::endl;
		return false;
	}

	thread = new std::thread(std::bind(&Velodyne::capture, this));

	std::cerr << "Velodyne capture successfully opened." << std::endl;
	return true;
}

void Velodyne::send(std::string& msg) {
	boost::asio::ip::udp::endpoint me;
	socket->send_to(boost::asio::buffer(msg, msg.size()), me);
}

bool Velodyne::isOpen()
{
	std::lock_guard<std::mutex> lock(mutex);
	return ((socket && socket->is_open()));
}

bool Velodyne::isRunning()
{
	// Returns True when Thread is Running or Queue is Not Empty
	std::lock_guard<std::mutex> lock(mutex);
	return (run || !queue.empty());
}

void Velodyne::capture() {
	run = true;
	unsigned char data[1206];
	boost::asio::ip::udp::endpoint sender;
	std::vector<Laser> * lasers = new std::vector<Laser>();
	double lastAzimuth = 0.0;

	while (socket->is_open() && ioService.stopped()) {
		//printf("*");
		boost::system::error_code error;
		const size_t length = socket->receive_from(boost::asio::buffer(data, sizeof(data)), sender, 0, error);
		if (error == boost::asio::error::eof) {
			break;
		}
		if (length != 1206) {
			continue;
		}

		// Convert to DataPacket Structure
		const DataPacket* packet = reinterpret_cast<const DataPacket*>(data);
		
		switch (this->model) {
		case ProductModelTable::HDL64:
			convertPacketHDL64(packet, lasers, lastAzimuth);
			break;
		case ProductModelTable::VLP16:
			convertPacketVLP16(packet, lasers, lastAzimuth);
			break;
		default:
			break;
		}
	}

	std::cout << "Capture thread ended." << std::endl;
	run = false;
	//size_t reply_length = this->socket.receive_from(boost::asio::buffer(packet, 1206), senderEndPoint);
}

void Velodyne::convertPacketHDL64(const DataPacket * packet, std::vector<Laser> *lasers, double &lastAzimuth) {
	//Check Data
	/*for (int i = 0; i < 12; i++) {
	printf("id %d: %u\n", i, packet->firingData[i].rotationalPosition);
	}*/

	// Fetch One Full Rotation
	// Calculate Interpolated Azimuth
	double interpolatedAzimuth = 0.0;
	if (packet->dataBlock[1].azimuth < packet->dataBlock[0].azimuth) {
		interpolatedAzimuth = ((packet->dataBlock[2].azimuth + 36000) - packet->dataBlock[0].azimuth) / 2.0;
	}
	else {
		interpolatedAzimuth = (packet->dataBlock[2].azimuth - packet->dataBlock[0].azimuth) / 2.0;
	}

	// Processing Packet
	//double lastAzimuth = 0.0;
	for (int blockIndex = 0; blockIndex < NDATABLOCK_PER_PKT; blockIndex++) { //FIRING_PER_PKT = 12
																				// Retrieve Firing Data
		int blockOffset;
		if ((blockIndex % 2) == 0) //upper block
			blockOffset = 0;
		else //lower block
			blockOffset = 32;

		const DataBlock dataBlock = packet->dataBlock[blockIndex];
		for (int laserIndex = 0; laserIndex < NLASER_PER_BLOCK; laserIndex++) { //LASER_PER_FIRING = 32
																				   // Retrieve Rotation Azimuth
			double azimuth = static_cast<double>(dataBlock.azimuth);

			// Interpolate Rotation Azimuth
			if (laserIndex >= MAX_NLASERS_HDL64)
			{
				azimuth += interpolatedAzimuth;
			}

			// Reset Rotation Azimuth
			if (azimuth >= 36000)
			{
				azimuth -= 36000;
			}
			// Complete Retrieve Capture One Rotation Data
			if (lastAzimuth > azimuth) {
				// Push One Rotation Data to Queue
				mutex.lock();
				//queue.push(lasers); //i only need the latest. change this to singular value?
				currentLasers = *lasers;
				mutex.unlock();
				lasers->clear();
			}
			Laser laser;
			if (isCalibrated) {
				laser.azimuth = azimuth / 100.0 - calibParams[laserIndex % MAX_NLASERS_HDL64 + blockOffset].rotCorrection;
				laser.vertical = calibParams[laserIndex % MAX_NLASERS_HDL64 + blockOffset].vertCorrection;
				if (dataBlock.laserReturns[laserIndex % MAX_NLASERS_HDL64].distance < 1.00) {
					laser.distance = 0.0;
				}
				else {
					laser.distance = dataBlock.laserReturns[laserIndex % MAX_NLASERS_HDL64].distance 
						+ calibParams[laserIndex % MAX_NLASERS_HDL64 + blockOffset].distCorrection * 10;
				}
			}
			else {
				laser.azimuth = azimuth / 100.0;
				laser.vertical = vertCorrHDL64[laserIndex % MAX_NLASERS_HDL64];
				if (dataBlock.laserReturns[laserIndex % MAX_NLASERS_HDL64].distance < 1.00) {
					laser.distance = 0.0;
				}
				else {
					laser.distance = dataBlock.laserReturns[laserIndex % MAX_NLASERS_HDL64].distance;
				}
			}
			
			//laser.vertical = vertAngle[laser_index % MAX_NUM_LASERS + blockOffset];
			
			laser.intensity = dataBlock.laserReturns[laserIndex % MAX_NLASERS_HDL64].intensity;
			laser.id = static_cast<unsigned char>(laserIndex % MAX_NLASERS_HDL64 + blockOffset);
			//laser.time = unixtime;
			laser.time = 0;

			lasers->push_back(laser);

			// Update Last Rotation Azimuth
			lastAzimuth = azimuth;
		}
	}
}

void Velodyne::convertPacketVLP16(const DataPacket * packet, std::vector<Laser> *lasers, double &lastAzimuth) {
	// Fetch One Full Rotation
	// Calculate interpolation offset of azimuth assuming constant rotation speed
	double interpolatedAzimuthOffset = 0.0;
	if (packet->dataBlock[1].azimuth < packet->dataBlock[0].azimuth) {
		interpolatedAzimuthOffset = ((static_cast<double>(packet->dataBlock[2].azimuth) + 36000.0) 
			- static_cast<double>(packet->dataBlock[0].azimuth)) / 2.0;
	}
	else {
		interpolatedAzimuthOffset = (static_cast<double>(packet->dataBlock[1].azimuth) 
			- static_cast<double>(packet->dataBlock[0].azimuth)) / 2.0;
	}

	// Processing Packet
	Laser laser;

	for (int blockIndex = 0; blockIndex < NDATABLOCK_PER_PKT; blockIndex++) { //NDATABLOCK_PER_PKT = 12

		const DataBlock dataBlock = packet->dataBlock[blockIndex];

		for (int laserIndex = 0; laserIndex < NLASER_PER_BLOCK; laserIndex++) { //NLASER_PER_BLOCK = 16 x 2 = 32
																				// Retrieve Rotation Azimuth
			double azimuth = static_cast<double>(dataBlock.azimuth); //even firing sequence

			// Interpolate Rotation Azimuth
			if (laserIndex >= MAX_NLASERS_VLP16) //odd firing sequence
			{
				azimuth += interpolatedAzimuthOffset;
			}

			// Reset azimuth angle
			if (azimuth >= 36000.0)
			{
				azimuth -= 36000.0;
			}

			//// Complete Retrieve Capture One Rotation Data
			if (lastAzimuth - azimuth > 30000) // TODO: should be lastAzimuth > azimuth
			{
				// Push One Rotation Data to Queue only once per half block
				if ((laserIndex == 0) || (laserIndex == 16)) {
					mutex.lock();
					currentLasers = *lasers;
					//isRefreshed = true;
					//std::cout << "Az: " << azimuth << " LA: " << lastAzimuth << "LL: " << lasers->size() << " " << laserIndex << std::endl;
					mutex.unlock();
					lasers->clear();
				}
			}

			if (isCalibrated) {
				laser.azimuth = azimuth / 100.0 - calibParams[laserIndex % MAX_NLASERS_VLP16].rotCorrection;
				laser.vertical = calibParams[laserIndex % MAX_NLASERS_VLP16].vertCorrection;
				if (dataBlock.laserReturns[laserIndex % MAX_NLASERS_VLP16].distance < 1.00) {
					laser.distance = 0.0;
				}
				else {
					laser.distance = dataBlock.laserReturns[laserIndex % MAX_NLASERS_VLP16].distance 
						+ calibParams[laserIndex % MAX_NLASERS_VLP16].distCorrection * 10;
				}
			}
			else {
				laser.azimuth = azimuth / 100.0;
				laser.vertical = vertCorrVLP16[laserIndex % MAX_NLASERS_VLP16];
				if (dataBlock.laserReturns[laserIndex % MAX_NLASERS_VLP16].distance < 1.00) {
					laser.distance = 0.0;
				}
				else {
					laser.distance = dataBlock.laserReturns[laserIndex % MAX_NLASERS_VLP16].distance;
				}
			}

			laser.intensity = dataBlock.laserReturns[laserIndex % MAX_NLASERS_VLP16].intensity;
			laser.id = static_cast<unsigned char>(laserIndex % MAX_NLASERS_VLP16);
			//laser.time = unixtime;
			laser.time = 0;

			lasers->push_back(laser);

			// Update Last Rotation Azimuth
			lastAzimuth = azimuth;
		}
	}
}

void Velodyne::retrieve(std::vector<Laser>& lasers, const bool sort = false)
{
	// Pop One Rotation Data from Queue
	if (mutex.try_lock()) {
		/*if (!queue.empty()) {
		lasers = queue.front();
		//if (sort) {
		//	std::sort(lasers.begin(), lasers.end());
		//}
		queue.pop();
		}*/
		//isRefreshed = false;
		lasers = currentLasers;
		mutex.unlock();
	}
}

// Operator Retrieve Capture Data with Sort
void Velodyne:: operator >> (std::vector<Laser>& lasers)
{
	// Retrieve Capture Data
	retrieve(lasers, false);
}

void Velodyne::close()
{
	std::lock_guard<std::mutex> lock(mutex);

	// Close Capturte Thread
	if (thread && thread->joinable()) {
		thread->detach();
		thread->~thread();
		delete thread;
		thread = nullptr;
	}

	if (socket && socket->is_open()) {
		socket->shutdown(boost::asio::ip::udp::socket::shutdown_both);
		socket->close();
		delete socket;
		socket = nullptr;
	}

	// Stop IO-Service
	if (ioService.stopped()) {
		ioService.stop();
		ioService.reset();
	}
}