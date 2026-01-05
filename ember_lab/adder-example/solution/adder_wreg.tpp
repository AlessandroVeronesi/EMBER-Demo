#ifndef __ADDER_WREG_TPP__
#define __ADDER_WREG_TPP__

//
// === adder_wreg === //
//

// PUBLIC
template<typename T>
const char* debug::adder_wreg<T>::id()
{
    return name.c_str();
}

template<typename T>
void debug::adder_wreg<T>::reset()
{
    outreg.reset();
    comb_add->reset();
}

template<typename T>
std::vector<ember::ISaboteur*> debug::adder_wreg<T>::getSaboteurs()
{
    std::vector<ember::ISaboteur*> regs;
    regs.push_back(dynamic_cast<ember::ISaboteur*>(&outreg));
    return regs;
}

template<typename T>
void debug::adder_wreg<T>::update()
{
    outreg.update();
    comb_add->update();
}

template<typename T>
void debug::adder_wreg<T>::eval()
{
    // --- Manual Topological Ordering Required --- //
    outreg.eval();
    comb_add->eval();
}


template<typename T>
debug::adder_wreg<T>::adder_wreg(const char* _id, const size_t bitwidth_a, const size_t bitwidth_b)
    :   inBwA(bitwidth_a),
        inBwB(bitwidth_b),
        outBw(std::max(inBwA,inBwB)+1),
        name(_id),
        A("A"),
        B("B"),
        C("C"),
        outreg("outreg", outBw)
{
    // Instantiate Sub Component
    comb_add = new debug::adder<T>("U", inBwA, inBwB);

    // External Self Bound Ports
    A.bind();
    B.bind();
    outreg.dout.bind();

    // Internal Connections
    comb_add->A.bind(A);
    comb_add->B.bind(B);

    outreg.din.bind(comb_add->C);
    C.bind(outreg.dout);
}

template<typename T>
debug::adder_wreg<T>::~adder_wreg()
{
    delete comb_add;
}

#endif