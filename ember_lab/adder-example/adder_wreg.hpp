#ifndef __ADDER_WREG_HPP__
#define __ADDER_WREG_HPP__

#include "ember.hpp"
#include "ember_saboteurs.hpp"

#include "adder.hpp"

namespace debug {

template<typename T>
class adder_wreg : public ember::IModule {
private:
    const std::string name;
    const size_t inBwA;
    const size_t inBwB;
    const size_t outBw;


    // Internal saboteurs
    ember::saboteur::ff<T> outreg;

    // Combinational Adder
    adder<T>* comb_add;

public:

    ember::inPort<T> A, B;
    ember::outPort<T> C;

    // Methods
    const char* id();
    void reset();
    void update();
    void eval();
    std::vector<ember::ISaboteur*> getSaboteurs();
    
    // CTOR
    adder_wreg(const char* _id, const size_t bitwidth_a, const size_t bitwidth_b);
    virtual ~adder_wreg();
};

}

#include "adder_wreg.tpp"

#endif