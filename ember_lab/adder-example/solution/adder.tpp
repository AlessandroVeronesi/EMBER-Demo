#ifndef __ADDER_TPP__
#define __ADDER_TPP__

//
// === adder === //
//
template<typename T>
const char* debug::adder<T>::id()
{
    return name.c_str();
}

template<typename T>
void debug::adder<T>::reset()
{
    // Do Noting: Combinational component
}

template<typename T>
std::vector<ember::ISaboteur*> debug::adder<T>::getSaboteurs()
{
    // Return empty list: no saboteurs inside
    std::vector<ember::ISaboteur*> regs;
    return regs;
}

template<typename T>
void debug::adder<T>::update()
{
    // Do Noting: Combinational component
}
    
template<typename T>
void debug::adder<T>::eval()
{
    C.write(A.read() + B.read());
}
    
template<typename T>
void debug::adder<T>::connect(ember::port<T>* _A, ember::port<T>* _B)
{
    A.bind(*_A);
    B.bind(*_B);
    C.bind();
}

template<typename T>
debug::adder<T>::adder(const char* _id, const size_t bitwidth_a, const size_t bitwidth_b)
    : inBwA(bitwidth_a),
    inBwB(bitwidth_b),
    outBw(std::max(inBwA,inBwB)+1),
    name(_id),
    A("A"),
    B("B"),
    C("C")
{
    // Nothing to do
}

template<typename T>
debug::adder<T>::~adder()
{
    // Nothing to do
}

#endif