#ifndef DAOZHA_INCLUDE_LOGGER_H_
#define DAOZHA_INCLUDE_LOGGER_H_
#include "header.h"

class Logger : public ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

#endif