#pragma once
#include <tuple>

// Credits:
// Based on
// tuple_slice: https://ngathanasiou.wordpress.com/2016/11/28/want-a-std-slice/
// reverse: https://stackoverflow.com/questions/25119048/reversing-a-c-tuple
// arithmetic & op: https://stackoverflow.com/questions/47209672/how-to-perform-tuple-arithmetic-in-c-c11-c17

namespace tuple_ext
{
  // Namespace the implementation details to stop them cluttering the top-level namespace
  namespace impl
  {
    template<typename... Types, std::size_t... Indexes>
    inline std::tuple<Types...> sum( std::tuple<Types...> lhs
                            , std::tuple<Types...> rhs
                            , std::index_sequence<Indexes...>&&)
    {
      return std::make_tuple(std::get<Indexes>(lhs) + std::get<Indexes>(rhs)...);
    }

    template<typename... Types, std::size_t... Indexes>
    inline std::tuple<Types...> sub( std::tuple<Types...> lhs
                            , std::tuple<Types...> rhs
                            , std::index_sequence<Indexes...>&&)
    {
      return std::make_tuple(std::get<Indexes>(lhs) - std::get<Indexes>(rhs)...);
    }

    template<typename... Types, std::size_t... Indexes>
    inline std::tuple<Types...> div( std::tuple<Types...>& lhs
                            , std::tuple<Types...>& rhs
                            , std::index_sequence<Indexes...>&&)
    {
      return std::make_tuple(std::get<Indexes>(lhs) / std::get<Indexes>(rhs)...);
    }

    template<typename... Types, std::size_t... Indexes>
    inline std::tuple<Types...> mul( std::tuple<Types...> lhs
                            , std::tuple<Types...> rhs
                            , std::index_sequence<Indexes...>&&)
    {
      return std::make_tuple(std::get<Indexes>(lhs) * std::get<Indexes>(rhs)...);
    }

    template<typename Op, typename Type, std::size_t... Indexes>
    inline auto op(Op f, Type t, std::index_sequence<Indexes...>&&)
    {
      return std::make_tuple(f(std::get<Indexes>(t))...);
    }

    template<typename Op, typename Type, std::size_t... Indexes>
    inline auto op(Op f, Type t1, Type t2, std::index_sequence<Indexes...>&&)
    {
      return std::make_tuple(f(std::get<Indexes>(t1), std::get<Indexes>(t2))...);
    }

    template <std::size_t Ofst, class Tuple, std::size_t... I>
    inline constexpr auto slice(Tuple&& t, std::index_sequence<I...>&&)
    {
      return std::forward_as_tuple(std::get<I + Ofst>(std::forward<Tuple>(t))...);
    }

    template<typename T, typename TT = typename std::remove_reference<T>::type, size_t... I>
    inline T reverse(T t, std::index_sequence<I...>)
    {
      return std::make_tuple(std::get<sizeof...(I)-1 - I>(t)...);
    }
  }  

  template<typename... Types>
  inline std::tuple<Types...> sum(std::tuple<Types...> lhs, std::tuple<Types...> rhs)
  {
    return impl::sum(lhs, rhs, std::make_index_sequence<sizeof...(Types)>{});
  }  

  template<typename... Types>
  inline std::tuple<Types...> sub(std::tuple<Types...> lhs, std::tuple<Types...> rhs)
  {
    return impl::sub(lhs, rhs, std::make_index_sequence<sizeof...(Types)>{});
  }

  template<typename... Types>
  inline std::tuple<Types...> mul(std::tuple<Types...> lhs, std::tuple<Types...> rhs)
  {
    return impl::mul(lhs, rhs, std::make_index_sequence<sizeof...(Types)>{});
  }

  template<typename... Types>
  inline std::tuple<Types...> div(std::tuple<Types...> lhs, std::tuple<Types...> rhs)
  {
    return impl::div(lhs, rhs, std::make_index_sequence<sizeof...(Types)>{});
  }

  // Perform op on two tuples, supports std functors
  template<typename Op, typename Type>
  inline auto op(Op f, Type t1, Type t2)
  {
    return impl::op(f, t1, t2, std::make_index_sequence<std::tuple_size_v<Type>>{});
  }

  // Single tuple op
  template<typename Op, typename Type>
  inline auto op(Op f, Type t)
  {
    return impl::op(f, t, std::make_index_sequence<std::tuple_size<std::remove_reference<Type>::type>::value>{});
  }

  template <std::size_t I1, std::size_t I2, class Cont>
  inline constexpr auto slice(Cont&& t)
  {
    static_assert(I2 >= I1, "invalid slice");
    static_assert(std::tuple_size<std::decay_t<Cont>>::value >= I2,
      "slice index out of bounds");

    return impl::slice<I1>(std::forward<Cont>(t), std::make_index_sequence<I2 - I1>{});
  }

  template<typename T, typename TT = typename std::remove_reference<T>::type>
  inline T reverse(T t)
  {
    return impl::reverse(t, std::make_index_sequence<std::tuple_size<TT>::value>());
  }

  // Handy CastTo tuple op
  template<typename T_Out>
  struct CastTo
  {
    template<typename T_In>
    inline constexpr auto operator()(T_In& in) const -> decltype(static_cast<T_Out>(in))
    {
      return static_cast<T_Out>(in);
    }
  };

  // Functor to subtract a pre-defined value from another value
  // Useful for subtracting a value from every item in a tuple
  template<int Value>
  struct ArbitrarySub
  {
    template<typename T_In>
    inline constexpr auto operator()(T_In& in) const -> decltype(in - Value)
    {
      return in - Value;
    }
  };

  template<int Value>
  struct ArbitraryAdd
  {
    template<typename T_In>
    inline constexpr auto operator()(T_In& in) const -> decltype(in + Value)
    {
      return in + Value;
    }
  };
}
