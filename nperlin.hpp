#pragma once
#include <array>
#include <tuple>
#include "tuple_ext.hpp"
#include <random>
#include <assert.h>
#include <cstdint>

static int FastFloor(float f) { return (f >= 0 ? (int)f : (int)f - 1); }
static float Lerp(float a, float b, float t) { return a + t * (b - a); }
static float InterpHermiteFunc(float t) { return t * t*(3 - 2 * t); }
static float InterpQuinticFunc(float t) { return t * t*t*(t*(t * 6 - 15) + 10); }

// Swaps bits a and b in the given value
inline constexpr uint64_t SwapBits(const uint64_t a, const uint64_t b, const uint64_t value)
{
  uint64_t tmp = (((value >> a) ^ (value >> b)) & ((1U << 1) - 1));
  return value ^ ((tmp << a) | (tmp << b));
}

// Src: https://stackoverflow.com/questions/17719674/c11-fast-constexpr-integer-powers
inline constexpr uint64_t ipow(const uint64_t base, uint64_t exp, const uint64_t result = 1) {
  return exp < 1 ? result : ipow(base*base, exp / 2, (exp % 2) ? result * base : result);
}

// Returns a std::array filled with the correct gradients for the given dimension, 
// gradients are not axis-aligned to reduce artifacts
template<size_t Dimensions>
constexpr std::array<float, static_cast<size_t>((Dimensions*ipow(2, (Dimensions - 1)))*Dimensions)> GradArray()
{
  constexpr size_t numberOfGradientsParts = static_cast<size_t>((Dimensions*ipow(2, (Dimensions - 1)))*Dimensions);
  constexpr size_t numberOfGradients = numberOfGradientsParts / Dimensions;
  constexpr size_t sizeOfSet = numberOfGradients / Dimensions;
  constexpr size_t numberOfSets = Dimensions;
  std::array<float, numberOfGradientsParts> arr{ 0 };

  for (unsigned int set = 0; set < numberOfSets; set++)
  {
    // This set tells us which value is to be 0 this set of gradients
    auto setGuide = SwapBits((numberOfSets - (set + 1)), (numberOfSets - 1), (ipow(2, numberOfSets - 1) - 1));

    for (unsigned int gradient = 0; gradient < sizeOfSet; gradient++)
    {
      int i = 0; // Counter for tracking position in gradient
      for (unsigned int dimension = 0; dimension < Dimensions; dimension++)
      {
        // If setGuide bit is 1, we check the i-th bit in gradient, if it is set we store -1.0f in arr, else 1.0f
        if (setGuide & (1Ui64 << dimension))
        {
          arr[(set*numberOfSets*sizeOfSet) + (gradient*Dimensions) + dimension] = (gradient & (1Ui64 << i)) ? -1.0f : 1.0f;
          i++;
          continue;
        }
        else // If setGuide bit is 0, we set this value to 0 and skip incrementing i, this creates the moving column of 0's in the gradients
        {
          arr[(set*numberOfSets*sizeOfSet) + (gradient*Dimensions) + dimension] = 0;
          // Do not increment i
          continue;
        }
      }
    }
  }
  return arr;
}

// 2D Special Case Partial Specialisation
template<>
inline constexpr std::array<float, static_cast<size_t>(8)> GradArray<2>()
{
  // These gradients aren't axis aligned and reduce chance of artifacts
  return std::array<float, 8>
  {
     1,  1,
     1, -1,
    -1,  1,
    -1, -1
  };
}

template<size_t Dimensions=4>
class Perlin
{
public:
  explicit Perlin(int64_t seed = 1337)
    : frequency(0.01f)
    , interp(Interp::Quintic)
  {
    SetSeed(seed); 
  }
  enum Interp { Linear, Hermite, Quintic };

  void SetSeed(uint64_t _seed);
  int64_t GetSeed() const { return seed; }

  void SetFrequency(float _frequency) { frequency = _frequency; }
  void SetInterp(Interp _interp) { interp = _interp; }

  template<class... Args> float Get(Args ...args) ;

private:
  static constexpr std::array<float, static_cast<size_t>((Dimensions*ipow(2, (Dimensions - 1)))*Dimensions)> GradientArray = GradArray<Dimensions>();
  static constexpr size_t numberOfGradients = static_cast<size_t>((Dimensions*ipow(2, (Dimensions - 1))));
  static constexpr size_t halfPerm =  Perlin<Dimensions>::numberOfGradients;
  std::array<uint64_t, numberOfGradients*2> perm;
  int64_t seed;
  float frequency;
  Interp interp;

  inline unsigned char lookupPerm(unsigned char offset) ;
  template<class... Dims> inline uint64_t lookupPerm(unsigned char offset, int head, Dims... tail) ;
  template<class... Args> inline uint64_t Index(unsigned char offset, Args... args) ;

  inline float lookupGrad(uint64_t lutPos, uint64_t step);
  template<class... Args> inline float lookupGrad(uint64_t lutPos, uint64_t step, float head, Args... tail);
  template<typename... Type1, typename... Type2> inline float GradCoord(unsigned char offset, std::tuple<Type1...>& iArgs, std::tuple<Type2...>& fArgs) ;
  
  // Collapses a tuple into a tuple half its size consisting of each two values lerp'd together
  template<typename Type, size_t... Indexes>
  inline auto LerpGrads(Type& grads, float interp, std::index_sequence<Indexes...>&&)
  {
    return std::make_tuple(Lerp(std::get<Indexes * 2>(grads), std::get<(Indexes * 2) + 1>(grads), interp)...);
  }

  // Sets up the real LerpGrads with an index sequence
  template<typename Type>
  inline auto LerpGrads(Type& grads, float interp)
  {
    return LerpGrads(grads, interp, std::make_index_sequence<std::tuple_size_v<Type> / 2>{});
  }

  // Exit function, takes a tuple of two floats and returns the lerp of those. 
  inline float LerpGrads(std::tuple<float, float>& grads, float interp)
  {
    return Lerp(std::get<0>(grads), std::get<1>(grads), interp);
  }

  // Recursive function to lerp all values together
  template <size_t Size, size_t Idx, class GradsType, class Interps>
  inline auto DoLerp(GradsType&& grads, Interps&& interps) {
    if constexpr (Idx == 1) {
      return LerpGrads(grads, std::get<Size - 1>(interps));
    }
    else {
      return DoLerp<Size, Idx - 1>(LerpGrads(grads, std::get<Size - Idx>(interps)), interps);
    }
  }

  template<typename Type1, typename Type2, size_t... Indexes>
  inline auto unpackGradCoords(unsigned char offset, Type1&& iArgs, Type2&& fArgs, std::index_sequence<Indexes...>&&)
  {
    return std::make_tuple(GradCoord(offset, std::get<Indexes>(iArgs), std::get<Indexes>(fArgs))...);
  }

  template<typename Type1, typename Type2>
  inline auto GradCoordUnpacker(unsigned char offset, Type1&& iArgs, Type2&& fArgs)
  {
    return unpackGradCoords(offset, std::forward<Type1>(iArgs), std::forward<Type2>(fArgs), std::make_index_sequence<std::tuple_size_v<Type1>>{});
  }

  template<class... Args> float SinglePerlin(unsigned char offset, Args... args) ;

  // Helper functor to select gradient coordinates
  struct FlipFlop
  {
    template<typename T>
    inline constexpr T operator()(size_t i, size_t step, T&& lhs, T&& rhs)
    {
      return (!(i & (static_cast<uint64_t>(1) << step)) ? lhs : rhs);
    }
  };

  // Helper functions for creating gradient cordinates
  template<typename Type, std::size_t... Indexes>
  inline auto getCoords_impl(Type& t1, Type& t2, size_t grad, std::index_sequence<Indexes...>&&)
  {
    auto f = FlipFlop{};
    return tuple_ext::reverse(std::make_tuple(f(grad, Indexes, std::get<Indexes>(t1), std::get<Indexes>(t2))...));
  }

  template<typename Type>
  inline auto getCoordsForGradient(Type&& t1, Type&& t2, size_t grad)
  {
    return getCoords_impl(std::forward<Type>(t1), std::forward<Type>(t2), grad, std::make_index_sequence<std::tuple_size<std::remove_reference_t<Type>>{}>{});
  }

  template<typename Type, std::size_t... GradIndexes>
  inline auto getCoordsForAllGradients_impl(Type&& t1, Type&& t2, std::index_sequence<GradIndexes...>&&)
  {
    return std::make_tuple(getCoordsForGradient(std::forward<Type>(t1), std::forward<Type>(t2), GradIndexes)...);
  }

  template<size_t numGrads, typename Type>
  inline auto getCoordsForAllGradients(Type&& t1, Type&& t2)
  {
    return getCoordsForAllGradients_impl(std::forward<Type>(t1), std::forward<Type>(t2), std::make_index_sequence<numGrads>{});
  }
};

template<size_t Dimensions> constexpr size_t Perlin<Dimensions>::numberOfGradients;
template<size_t Dimensions> constexpr std::array<float, static_cast<size_t>((Dimensions*ipow(2, (Dimensions - 1)))*Dimensions)> Perlin<Dimensions>::GradientArray;
template<size_t Dimensions> constexpr size_t Perlin<Dimensions>::halfPerm;

template<size_t Dimensions>
inline void Perlin<Dimensions>::SetSeed(uint64_t _seed)
{
  seed = _seed;

  std::mt19937_64 gen(seed);

  perm = { 0 };

  for (uint64_t i = 0; i < halfPerm; i++)
    perm[i] = i;

  for (uint64_t j = 0; j < halfPerm; j++)
  {
    std::uniform_int_distribution<uint64_t> dist(0, halfPerm - j);
    uint64_t k = dist(gen) + j;
    uint64_t l = perm[j];
    perm[j] = perm[j + halfPerm] = perm[k];
    perm[k] = l;
  }
}

template<size_t Dimensions>
template<class... Args>
inline float Perlin<Dimensions>::Get(Args ...args)
{
  //assert(sizeof...(args) == Dimensions);
  static_assert(sizeof...(args) == Dimensions, "Wrong number of arguments passed to Get!");

  ((args *= frequency), ...);

  return SinglePerlin(0, args...);
}

template<size_t Dimensions>
inline unsigned char Perlin<Dimensions>::lookupPerm(unsigned char offset)
{
  return offset;
}

template<size_t Dimensions>
template<class... Dims>
inline uint64_t Perlin<Dimensions>::lookupPerm(unsigned char offset, int head, Dims... tail)
{
  return perm[(head % halfPerm) + lookupPerm(offset, tail...)];
}

template<size_t Dimensions>
template<class... Args>
inline uint64_t Perlin<Dimensions>::Index(unsigned char offset, Args... args)
{
  assert(sizeof...(args) == Dimensions);

  return lookupPerm(offset, args...);
}

template<size_t Dimensions>
inline float Perlin<Dimensions>::lookupGrad(uint64_t lutPos, uint64_t step)
{
  return 0;
}

template<size_t Dimensions>
template<class... Args>
inline float Perlin<Dimensions>::lookupGrad(uint64_t lutPos, uint64_t step, float head, Args... tail)
{
  return (head * GradientArray[lutPos + step]) + lookupGrad(lutPos, step+1, tail...);
}

template<size_t Dimensions>
template<typename... Type1, typename... Type2>
inline float Perlin<Dimensions>::GradCoord(unsigned char offset, std::tuple<Type1...>& iArgs, std::tuple<Type2...>& fArgs)
{
  uint64_t lutPos = std::apply([&](auto ...x) {return Index(x...); }, std::tuple_cat(std::make_tuple(offset), iArgs)) * Dimensions;
  uint64_t step = 0;

  return std::apply([&](auto ...x) { return lookupGrad(x...); }, std::tuple_cat(std::make_tuple(lutPos, step), fArgs));
}

template<size_t Dimensions>
template<class... Args>
float Perlin<Dimensions>::SinglePerlin(unsigned char offset, Args ...args)
{
  assert(sizeof...(args) == Dimensions);
  
  auto argsIn = std::make_tuple(args...);
  auto argsFloored = std::apply([](auto ...x) {return std::make_tuple(FastFloor(x)...); }, argsIn);
  auto argsPlusOne = tuple_ext::op(tuple_ext::ArbitraryAdd<1>{}, argsFloored);

  std::tuple<Args...> interps;
  auto linear = tuple_ext::sub(argsIn, tuple_ext::op(tuple_ext::CastTo<float>{}, argsFloored));
  switch (interp)
  {
  case Interp::Linear:
  {
    interps = linear;
  }
  break;
  case Interp::Hermite:
  {
    interps = std::apply([&](auto ...x) {return std::make_tuple(InterpHermiteFunc(x)...); }, linear);
  }
  break;
  case Interp::Quintic:
  {
    interps = std::apply([&](auto ...x) {return std::make_tuple(InterpQuinticFunc(x)...); }, linear);
  }
  break;
  default: abort();
  }

  auto d0 = linear;
  auto d1 = tuple_ext::op(tuple_ext::CastTo<float>{}, tuple_ext::op(tuple_ext::ArbitrarySub<1>{}, d0));;
  auto gradCoordsA = getCoordsForAllGradients<ipow(2, Dimensions)>(argsFloored, argsPlusOne); // x0, y1, etc...
  auto gradCoordsB = getCoordsForAllGradients<ipow(2, Dimensions)>(d0, d1); // xd0, yd1, etc...    

  auto grads = GradCoordUnpacker(offset, std::move(gradCoordsA), std::move(gradCoordsB));

  // Recursively interpolate values per dimensions
  constexpr size_t size = std::tuple_size_v<decltype(interps)>;
  return DoLerp<size, size>(grads, interps);
}

// 1D specialisation
//  return Lerp(GradCoord1D(offset, x0, xd0), GradCoord1D(offset, x1, xd1), xs);

