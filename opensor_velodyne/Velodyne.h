#ifndef VELODYNE_H
#define VELODYNE_H

#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/system/config.hpp>
#include <mutex>
#include <thread>
#include <queue>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>

#include <pcap.h>

#define BOOST_LIB_PATH "D:/dev/staticlib/boost_1_67_0/stage/lib/"
#pragma comment(lib, BOOST_LIB_PATH "libboost_system-vc141-mt-x64-1_67.lib")
#pragma comment(lib, "D:/dev/lib64/Packet.lib")
#pragma comment(lib, "D:/dev/lib64/wpcap.lib")

struct CalibParams {
	unsigned char laserId;
	unsigned int blockId;
	double rotCorrection;
	double vertCorrection;
	double distCorrection;
	double distCorrectionX;
	double distCorrectionY;
	double vertOffsetCorrection;
	double horizOffsetCorrection;
	double focalDistance;
	double focalSlope;
	unsigned char minIntensity;
	unsigned char maxIntensity;
};

struct Laser
{
	double azimuth;
	double vertical;
	unsigned short distance;
	unsigned char intensity;
	unsigned char id;
	long long time;

	const bool operator < (const struct Laser& laser) {
		if (azimuth == laser.azimuth) {
			return id < laser.id;
		}
		else {
			return azimuth < laser.azimuth;
		}
	}
};

typedef uint8_t ReturnMode;
typedef uint8_t ProductModel;

enum ReturnModeTable : uint8_t {
	STRONGEST = 0x37,
	LASTRETURN = 0x38,
	DUALRETURN = 0x39
};

enum ProductModelTable : uint8_t {
	HDL64, //sensortype = i don't know
	HDL32E = 0x21,
	VLP16 = 0x22, //sensortype = 0x22 
	PUCKLITE = 0x22,
	PUCKHIRES = 0x24,
	VLP32C = 0x28,
	VELARRAY = 0x31,
	VLS128 = 0x63
};

enum CaptureMode {
	UDP, TCP, PCAP
};

class Velodyne {
public:
	// Reading UDP broadcast
	Velodyne(boost::asio::ip::address& address, unsigned short port);
	Velodyne(const char *address_str, unsigned short port);

	// Reading PCAP
	Velodyne(const std::string& filename);

	//Reading TCP (from velodyne dedicated server of MRBUS)
	// TODO

	~Velodyne();

	CaptureMode captureMode;

	// Common
	boost::asio::io_service ioService;
	boost::asio::ip::udp::socket* socket = nullptr;
	boost::asio::ip::address address;
	unsigned short port = 2368;
	pcap_t* pcap = nullptr;
	std::string filename = "";

	std::mutex mutex;
	std::thread* thread = nullptr;
	std::atomic_bool run = { false };
	std::queue<std::vector<Laser>> queue;

	bool isCalibrated = false;

	bool open(boost::asio::ip::address& address, const unsigned short port);
	bool open(const std::string& filename);

	void send(std::string& msg);
	void capture();
	void capturePCAP();

	ProductModel model = VLP16; //default to VLP16 model
	bool isOpen();
	bool isRunning();
	void close();
	int laserCount;
	std::atomic_bool isRefreshed = { false };

#pragma pack(push, 1)
	typedef struct LaserReturn
	{
		uint16_t distance;
		uint8_t intensity;
	} LaserReturn;
#pragma pack(pop)

	std::vector<Laser> currentLasers;
	std::vector<CalibParams> calibParams;
	static const int NDATABLOCK_PER_PKT = 12;
	static const int NLASER_PER_BLOCK = 32;

	struct DataBlock
	{
		uint16_t flag;
		uint16_t azimuth;
		LaserReturn laserReturns[NLASER_PER_BLOCK];
	};

	struct DataPacket
	{
		DataBlock dataBlock[NDATABLOCK_PER_PKT];
		uint32_t gpsTimestamp;
		ReturnMode mode;
		ProductModel productModel;
	};

	// HDL64
	int MAX_NLASERS_HDL64 = 64;
	void convertPacketHDL64(const DataPacket* packet, std::vector<Laser> *lasers, double &lastAzimuth);

	// VLP16
	int MAX_NLASERS_VLP16 = 16;
	void convertPacketVLP16(const DataPacket* packet, std::vector<Laser> *lasers, double &lastAzimuth);
	

	void calibFileRead(const char *filename);

	//std::vector<double> lut;
	const std::vector<double> vertCorrHDL64 = { -8.356335,-7.954661, 2.790216, 3.152316, -7.506155, -6.95288, -10.464894, -10.033926,
		-6.398299, -6.004734, -9.533476, -9.031554, -2.254343, -1.868344, -5.239164, -4.855656,
		-1.318304, -0.849989, -4.40186, -3.865905, -0.311289, 0.08692, -3.340945, -2.885521,
		3.899047, 4.271932, 0.848147, 1.257932, 4.818941, 5.260565, 1.726097, 2.264198,
		-22.277325, -21.935509, -11.216363, -10.730993, -21.460285, -20.94109, -24.506605, -24.039431,
		-20.39769, -19.767496, -23.470364, -22.975813, -16.187433, -15.819491, -19.163635, -18.765581,
		-15.40667, -14.860983, -18.481606, -17.921432, -14.235523, -13.695058, -17.347025, -16.71529,
		-10.039779, -9.675978, -13.085444, -12.617874, -9.311386, -8.740197, -12.305205, -11.745022
	}; //Actually this is the vertical angle

	const std::vector<double> vertOffsetCorrVLP16 = { 11.2, -0.7, 9.7, -2.2, 8.1, -3.7, 6.6,
		-5.1, 5.1, -6.6, 3.7, -8.1, 2.2, -9.7, 0.7, -11.2
	};

	const std::vector<double> vertCorrVLP16 = { -15, 1, -13, 3, -11, 5, -9,
		7, -7, 9, -5, 11, -3, 13, -1, 15
	};

	std::vector<double> vertAngle;

	const std::vector<double> rotFarCorr = {};
	const std::vector<double> distFarCorr = {};
	const std::vector<double> vertOffsetCorr = {};
	const std::vector<double> horizOffsetCorr = {};



	




	//boost::asio::ip::udp::endpoint endPoint;
	//boost::asio::ip::udp::endpoint senderEndPoint;

	//char *packet;

	void retrieve(std::vector<Laser>& lasers, const bool sort);
	// Operator Retrieve Capture Data with Sort
	void operator >> (std::vector<Laser>& lasers);
};

#endif