#include <string>
#include <vector>
#include <iostream>
#include <chrono>
#include <cassert>

class TimeLogger
{
public:
    void startEvent(const std::string &stageName)
    {
        startTime = std::chrono::high_resolution_clock::now();
        stageNames.push_back(stageName);
    }
    void endEvent()
    {
        endTime = std::chrono::high_resolution_clock::now();
        auto execTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        stageDurations.push_back(execTime.count());
    }

    void printTimings()
    {
        assert(stageNames.size() == stageDurations.size());
        std::cout << "___________Execution time profiling___________" << std::endl;
        for (size_t eventIdx = 0; eventIdx < stageDurations.size(); ++eventIdx)
        {
            std::cout << stageNames[eventIdx] << ":  " << stageDurations[eventIdx] << " ms." << std::endl;
        }
    }

private:
    std::vector<std::string> stageNames;
    std::vector<double> stageDurations;

    std::chrono::high_resolution_clock::time_point startTime;
    std::chrono::high_resolution_clock::time_point endTime;
};