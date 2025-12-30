
#include <iostream>
#include <bitset>

#include "ember.hpp"

// Design Under Test
#include "adder_wreg.hpp"

#define TESTSIZE 100
#define BITWIDTH 8
#define SIMTIME  5

int main(int argc, char* argv[])
{
    // --- DUT --- //
    debug::adder_wreg<ember::int32_t>* dut;
    dut = new debug::adder_wreg<ember::int32_t>("ADD", BITWIDTH, BITWIDTH);    // Create DUT

    // --- Warm-Up Time --- //
    ember::time_t<long unsigned> dut_warmup = 1;                     // Expected delay

    // --- Saboteurs List --- //
    std::vector<ember::ISaboteur*> reglist = dut->getSaboteurs();    // Get DUT saboteurs


    // --- Testbench Variables --- //
    ember::int32_t input_a, input_b, output_c;
    ember::int32_t maxRange = (0x1 << (BITWIDTH-1)) -1;
    ember::int32_t minRange = -(0x1 << (BITWIDTH-1));


    // --- Core Testbench Routine --- //
    for(size_t testit=0; testit<TESTSIZE; testit++) {

        // --- Generate Random Inputs --- //
        input_a = ember::math::random::uniform<ember::int32_t>(minRange, maxRange);
        input_b = ember::math::random::uniform<ember::int32_t>(minRange, maxRange);
        
        // --- Expected Output --- //
        output_c = input_a + input_b;

        // --- Clear DUT Saboteurs State --- //
        for(size_t it=0; it<reglist.size(); it++) {
            reglist[it]->clearAllMasks();
        }

        // --- Generate SEU Fault --- //
        ember::time_t<long unsigned> itime = static_cast<long unsigned>(ember::math::random::uniform(int(dut_warmup.getSimTime()+1), int(SIMTIME-1)));
        size_t pos = ember::math::random::uniform((size_t)0, reglist.size()); // ember rand int uniform is in [start, end)
        ember::ISaboteur* loc = reglist[pos];
        ember::fault::seu_t<long unsigned> mySeu(itime, loc);
        std::cout << ">> Generated Fault: " << mySeu << std::endl;


        // --- Core Simulation Routine --- //
        for(ember::time_t<long unsigned> tick = 0; tick < SIMTIME; tick++) {

            // --- DUT Update --- //
            dut->A.write(input_a);
            dut->B.write(input_b);
            dut->update();
            dut->eval();

            // --- Inject SEU Fault (at the right time) --- //
            if (mySeu.time() == tick) {
                std::cout << ">> SimTime = " << tick << ": \033[1;31minjecting fault " << mySeu << "\033[0m" << std::endl;
                mySeu.location()->genFaultMask(mySeu.fmodel());
                mySeu.location()->applyAllFaults();
            }


            // --- Monitor Display --- //
            std::cout << ">> SimTime = " << tick << " inputs(" << input_a << ", " << input_b << ")" << std::endl;
            if(tick >= dut_warmup)
            {
                if(dut->C.read() == output_c) {
                    ember::int32_t temp = dut->C.read();
                    std::cout << ">> Output(" << temp << ")" << std::endl;
                }
                else {
                    ember::int32_t temp = dut->C.read();
                    std::bitset<BITWIDTH+1> golden(output_c), error(temp);
                    std::cout << ">> \033[1;31mOutput(" << temp << ") / Expected(" << output_c << ")\033[0m" << std::endl;
                    std::cout << ">> Expected(" << golden << ")" << std::endl;
                    std::cout << ">> Received(" << error << ")" << std::endl;
#ifndef RUNONFAILS
                    std::cout << "##########################################################################################" << std::endl;
                    std::cout << "###" << std::endl;
                    std::cout << "### TEST " << testit << " FAILED" << std::endl;
                    std::cout << "###" << std::endl;
                    std::cout << "##########################################################################################" << std::endl;
                    return -1;
#endif
                }
            }
            else {
                std::cout << std::endl;
            }
        }

        std::cout << "##########################################################################################" << std::endl;
        std::cout << "###" << std::endl;
        std::cout << "### TEST " << testit << " DONE" << std::endl;
        std::cout << "###" << std::endl;
        std::cout << "##########################################################################################" << std::endl;
    }

    std::cout << "=> ALL TESTS DONE" << std::endl;

    delete dut;

    return 0;
}
