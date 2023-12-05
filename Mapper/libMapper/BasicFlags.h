// flags for debugging purposes etc
#define _DEBUG

#ifdef _DEBUG // or #ifndef NDEBUG
#define LOG_MSG(msg) std::cout << msg << std::endl // Or simply LOG_MSG(msg) printf(msg)
#else
#define LOG_MSG(msg)                     // Or LOG_MSG(msg)
#endif