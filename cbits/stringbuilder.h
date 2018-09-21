// MIT License
//
// Copyright (c) 2017 Mariusz Łapiński
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <new>
#include <array>
#include <deque>
#include <sstream>
#include <memory>
#include <numeric>
#include <utility>
#include <assert.h>
#include <type_traits>
#ifdef __GNUC__
#include <experimental/string_view>
#if defined(__cpp_lib_experimental_string_view) && __cpp_lib_experimental_string_view >= 201411
namespace std {
    template<typename... AnyT> using basic_string_view = experimental::basic_string_view<AnyT...>;
    template<typename... AnyT> using void_t = __void_t<AnyT...>;
}
#endif
#else
#include <string_view>
#endif

#ifdef __GNUC__
#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)
#define noinline        __attribute__((noinline))
#else
#define likely(x)       (x)
#define unlikely(x)     (x)
#define noinline        __declspec(noinline)
#endif

#ifdef WIN32
#include <intrin.h>
#endif

void prefetchWrite(const void* p)
{
#ifdef __GNUC__
    __builtin_prefetch(p, 1);
#else
    _mm_prefetch((const char*)p, _MM_HINT_T0);
#endif
}


template<size_t ExpectedSize, typename StringT>
struct sized_str_t
{
    StringT str;
};

template<size_t ExpectedSize, typename StringT>
auto sized_str(StringT&& str)
{
    return sized_str_t<ExpectedSize, StringT>{ std::forward<StringT>(str) };
}


// Appender is an utility class for encoding various kinds of objects (integers) and their propagation to stringbuilder or inplace_stringbuilder.
//

// Unless there are suitable converters, make use of to_string() to stringify the object.
template<typename SB, typename T, typename Enable = void>
struct sb_appender {
    void operator()(SB& sb, const T& v) const {
        using namespace std; // ADL-aware
        sb.append(to_string(v));
    }
};

template<typename SB, size_t ExpectedSize, typename StringT>
struct sb_appender<SB, sized_str_t<ExpectedSize, StringT>>
{
    void operator()(SB& sb, const sized_str_t<ExpectedSize, StringT>& sizedStr) const
    {
        sb.append(sizedStr.str);
    }
};

template<typename SB>
struct sb_appender<SB, const typename SB::char_type*>
{
    void operator()(SB& sb, const typename SB::char_type* str) const
    {
        sb.append_c_str(str);
    }
};


// Provides means for building size-delimited strings in-place (without heap allocations).
// Object of this class occupies fixed size (specified at compile time) and allows appending portions of strings unless there is space available.
// In debug mode appending ensures that the built string does not exceed the available space.
// In release mode no such checks are made thus dangerous memory corruption may occur if used incorrectly.
//
template<typename CharT,
    size_t MaxSize,
    bool Forward,
    typename Traits>
class basic_inplace_stringbuilder
{
    static_assert(MaxSize > 0, "MaxSize must be greater than zero");

public:
    using traits_type = Traits;
    using char_type = CharT;
    using value_type = char_type;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;
    using reference = char_type&;
    using const_reference = const char_type&;
    using pointer = char_type*;
    using const_pointer = const char_type*;

    size_type size() const noexcept { return consumed; }
    size_type length() const noexcept { return size(); }

    basic_inplace_stringbuilder& append(char_type ch) noexcept
    {
        assert(ch != '\0');
        assert(consumed < MaxSize);
        if /*constexpr*/ (Forward) {
            data_[consumed++] = ch;
        } else {
            data_[MaxSize - (++consumed)] = ch;
        }
        return *this;
    }

    basic_inplace_stringbuilder& append(size_type count, char_type ch) noexcept
    {
        assert(ch != '\0');
        assert(consumed + count <= MaxSize);
        if /*constexpr*/ (Forward) {
            while (count-- > 0) data_[consumed++] = ch;
        } else {
            while (count-- > 0) data_[MaxSize - (++consumed)] = ch;
        }
        return *this;
    }

    template<size_type N>
    basic_inplace_stringbuilder& append(const char_type(&str)[N]) noexcept
    {
        return append(str, N-1);
    }

    template<size_type N>
    basic_inplace_stringbuilder& append(const std::array<char_type, N>& arr) noexcept
    {
        return append(arr.data(), N);
    }

    basic_inplace_stringbuilder& append(const char_type* str, size_type size) noexcept
    {
        assert(consumed + size <= MaxSize);
        if (Forward) {
            Traits::copy(data_.data() + consumed, str, size);
        } else {
            Traits::copy(data_.data() + MaxSize - size - consumed, str, size);
        }
        consumed += size;
        return *this;
    }

    basic_inplace_stringbuilder& append_c_str(const char_type* str, size_type size) noexcept
    {
        return append(str, size);
    }

    basic_inplace_stringbuilder& append_c_str(const char_type* str) noexcept
    {
        return append(str, Traits::length(str));
    }

    basic_inplace_stringbuilder& append_c_str_progressive(const char_type* str) noexcept
    {
        assert(Forward);
        while (*str != 0)
        {
            assert(consumed <= MaxSize);
            data_[consumed++] = *(str++);
        }
        return *this;
    }

    template<typename OtherTraits, typename OtherAlloc>
    basic_inplace_stringbuilder& append(const std::basic_string<char_type, OtherTraits, OtherAlloc>& str) noexcept
    {
        return append(str.data(), str.size());
    }

    template<typename OtherTraits>
    basic_inplace_stringbuilder& append(const std::basic_string_view<char_type, OtherTraits>& sv) noexcept
    {
        return append(sv.data(), sv.size());
    }

    template<size_type OtherMaxSize, bool OtherForward, typename OtherTraits>
    basic_inplace_stringbuilder& append(const basic_inplace_stringbuilder<char_type, OtherMaxSize, OtherForward, OtherTraits>& ss) noexcept
    {
        return append(ss.str_view());
    }

    template<typename T>
    basic_inplace_stringbuilder& append(const T& v)
    {
        sb_appender<basic_inplace_stringbuilder, T>{}(*this, v);
        return *this;
    }

    template<typename AnyT>
    basic_inplace_stringbuilder& operator<<(AnyT&& any)
    {
        return append(std::forward<AnyT>(any));
    }

    template<typename AnyT>
    basic_inplace_stringbuilder& append_many(AnyT&& any)
    {
        return append(std::forward<AnyT>(any));
    }

    template<typename AnyT1, typename... AnyTX>
    basic_inplace_stringbuilder& append_many(AnyT1&& any1, AnyTX&&... anyX)
    {
        return append(std::forward<AnyT1>(any1)).append_many(std::forward<AnyTX>(anyX)...);
    }

    char_type* data() noexcept
    {
        return data_.data() + (Forward ? 0 : MaxSize - consumed);
    }

    const char_type* data() const noexcept
    {
        return data_.data() + (Forward ? 0 : MaxSize - consumed);
    }

    const char_type* c_str() const noexcept
    {
        // Placing '\0' at the end of string is a kind of lazy evaluation and is acceptable also for const objects.
        const_cast<basic_inplace_stringbuilder*>(this)->placeNullTerminator();
        return data();
    }

    std::basic_string<char_type> str() const
    {
        assert(consumed <= MaxSize);
        const auto b = data_.cbegin();
        if /*constexpr*/ (Forward) {
            return {b, b + consumed};
        } else {
            return {b + MaxSize - consumed, b + MaxSize};
        }
    }

    std::basic_string_view<char_type, traits_type> str_view() const noexcept
    {
        return {data(), size()};
    }

private:
    void placeNullTerminator() noexcept
    {
        assert(consumed <= MaxSize);
        const_cast<char_type&>(data_[Forward ? consumed : MaxSize]) = '\0';
    }

private:
    size_type consumed = 0;
    std::array<char_type, MaxSize + 1> data_; // Last character is reserved for '\0'.
};

namespace detail
{
    // The following code is based on ebo_helper explained in this talk:
    // https://youtu.be/hHQS-Q7aMzg?t=3039
    //
    template<typename OrigAlloc, bool UseEbo = !std::is_final<OrigAlloc>::value && std::is_empty<OrigAlloc>::value>
    struct raw_alloc_provider;

    template<typename OrigAlloc>
    struct raw_alloc_provider<OrigAlloc, true> : private std::allocator_traits<OrigAlloc>::template rebind_alloc<uint8_t>
    {
        using AllocRebound = typename std::allocator_traits<OrigAlloc>::template rebind_alloc<uint8_t>;

        template<typename OtherAlloc> constexpr explicit raw_alloc_provider(OtherAlloc&& otherAlloc) : AllocRebound{ std::forward<AllocRebound>(otherAlloc) } {}
        constexpr AllocRebound& get_rebound_allocator() { return *this; }
        constexpr OrigAlloc get_original_allocator() const { return OrigAlloc{ *this }; }
    };

    template<typename OrigAlloc>
    struct raw_alloc_provider<OrigAlloc, false>
    {
        using AllocRebound = typename std::allocator_traits<OrigAlloc>::template rebind_alloc<uint8_t>;

        template<typename OtherAlloc> constexpr explicit raw_alloc_provider(OtherAlloc&& otherAlloc) : alloc_rebound{ std::forward<AllocRebound>(otherAlloc) } {}
        constexpr AllocRebound& get_rebound_allocator() { return alloc_rebound; }
        constexpr OrigAlloc get_original_allocator() const { return OrigAlloc{ alloc_rebound }; }
    private:
        AllocRebound alloc_rebound;
    };


    template<typename CharT>
    struct Chunk;

    template<typename CharT>
    struct ChunkHeader
    {
        Chunk<CharT>* next;
        size_t consumed;
        size_t reserved;
    };

    template<typename CharT>
    struct Chunk : public ChunkHeader<CharT>
    {
        CharT data[1]; // In practice there are ChunkHeader::reserved of characters in this array.

        Chunk(size_t reserve) : ChunkHeader<CharT>{nullptr, size_t{0}, reserve} { }
    };

    template<typename CharT, int DataLength>
    struct ChunkInPlace : public ChunkHeader<CharT>
    {
        std::array<CharT, DataLength> data;

        ChunkInPlace() : ChunkHeader<CharT>{nullptr, 0, DataLength} { }
    };

    template<typename CharT>
    struct ChunkInPlace<CharT, 0> : public ChunkHeader<CharT>
    {
        ChunkInPlace() : ChunkHeader<CharT>{nullptr, 0, 0} { }
    };
}


// Provides means for efficient construction of strings.
// Object of this class occupies fixed size (specified at compile time) and allows appending portions of strings.
// If the available space gets exhausted, new chunks of memory are allocated on the heap.
//
template<typename Char,
    size_t InPlaceSize,
    typename Traits,
    typename AllocOrig>
class basic_stringbuilder : private detail::raw_alloc_provider<AllocOrig>
{
    using AllocProvider = detail::raw_alloc_provider<AllocOrig>;
    using Alloc = typename AllocProvider::AllocRebound;
    using AllocTraits = std::allocator_traits<Alloc>;

public:
    using traits_type = Traits;
    using char_type = Char;
    using value_type = char_type;
    using allocator_type = AllocOrig;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;
    using reference = char_type&;
    using const_reference = const char_type&;
    using pointer = char_type*;
    using const_pointer = const char_type*;

private:
    using Chunk = detail::Chunk<char_type>;
    using ChunkHeader = detail::ChunkHeader<char_type>;
    template<int N> using ChunkInPlace = detail::ChunkInPlace<char_type, N>;

public:
    AllocOrig get_allocator() const noexcept { return AllocProvider::get_original_allocator(); }

    template<typename AllocOther = Alloc>
    basic_stringbuilder(AllocOther&& allocOther = AllocOther{}) noexcept : AllocProvider{std::forward<AllocOther>(allocOther)} {}

    basic_stringbuilder(const basic_stringbuilder&) = delete;

    basic_stringbuilder(basic_stringbuilder&& other) noexcept :
        AllocProvider{other.get_allocator()},
        headChunkInPlace{other.headChunkInPlace},
        tailChunk{other.tailChunk}
    {
        other.headChunkInPlace.next = nullptr;
    }

    ~basic_stringbuilder()
    {
        Chunk* nextChunk = headChunk()->next;
        for (auto chunk = nextChunk; chunk != nullptr; chunk = nextChunk)
        {
            nextChunk = chunk->next;
            //AllocTraits::destroy...?
            AllocTraits::deallocate(AllocProvider::get_rebound_allocator(), reinterpret_cast<typename AllocTraits::pointer>(chunk), sizeof(ChunkHeader) + chunk->reserved);
        }
    }

    size_type size() const noexcept
    {
        size_type size = 0;
        const auto* chunk = headChunk();
        do {
            size += chunk->consumed;
            chunk = chunk->next;
        } while (chunk != nullptr);
        return size;
    }

    size_type length() const noexcept { return size(); }

    void reserve(size_type size)
    {
        for (Chunk* chunk = tailChunk; size > chunk->reserved - chunk->consumed; chunk = chunk->next)
        {
            size -= chunk->reserved - chunk->consumed;
            assert(size > 0);
            if (chunk->next == nullptr) {
                chunk->next = allocChunk(size);
            }
        }
    }

    basic_stringbuilder& append(char_type ch)
    {
        assert(ch != '\0');
        claimOne() = ch;
        return *this;
    }

    basic_stringbuilder& append(size_type count, char_type ch)
    {
        assert(ch != '\0');
        for (auto left = count; left > 0;) {
            const auto claimed = claim(1, left);
            for (size_type i = 0; i < claimed.second; ++i) {
                claimed.first[i] = ch;
            }
            left -= claimed.second;
        }
        return *this;
    }

    template<size_type StrSizeWith0>
    basic_stringbuilder& append(const char_type(&str)[StrSizeWith0])
    {
        assert(str[StrSizeWith0-1] == 0);
        return append(str, StrSizeWith0 - 1);
    }

    template<size_type N>
    basic_stringbuilder& append(const std::array<char_type, N>& arr)
    {
        return append(arr.data(), N);
    }

    basic_stringbuilder& append(const char_type* str, size_type size)
    {
        Traits::copy(claim(size), str, size);
        return *this;
    }

    basic_stringbuilder& append_c_str(const char_type* str, size_type size)
    {
        return append(str, size);
    }

    template<bool Prefetch = false>
    basic_stringbuilder& append_c_str(const char_type* str)
    {
        if (Prefetch) prefetchWrite(&tailChunk->data[tailChunk->consumed]);
        return append(str, Traits::length(str));
    }

    basic_stringbuilder& append_c_str_progressive(const char_type* str)
    {
        while (true) {
            auto claimed = claim(1, 64);
            auto dst = claimed.first;
            const auto dstEnd = dst + claimed.second;
            while (dst < dstEnd) {
                if (*str == 0) {
                    reclaim(dstEnd - dst);
                    return *this;
                }
                *(dst++) = *(str++);
            }
        }
        assert(false);
        return *this;
    }

    template<typename OtherTraits, typename OtherAlloc>
    basic_stringbuilder& append(const std::basic_string<char_type, OtherTraits, OtherAlloc>& str)
    {
        return append(str.data(), str.size());
    }

    template<typename OtherTraits>
    basic_stringbuilder& append(const std::basic_string_view<char_type, OtherTraits>& sv)
    {
        return append(sv.data(), sv.size());
    }

    template<size_type OtherMaxSize, bool OtherForward, typename OtherTraits>
    basic_stringbuilder& append(const basic_inplace_stringbuilder<char_type, OtherMaxSize, OtherForward, OtherTraits>& sb)
    {
        return append(sb.str_view());
    }

    template<size_type OtherInPlaceSize, typename OtherTraits, typename OtherAlloc>
    basic_stringbuilder& append(const basic_stringbuilder<char_type, OtherInPlaceSize, OtherTraits, OtherAlloc>& sb)
    {
        size_type size = sb.size();
        reserve(size);

        const Chunk* chunk = sb.headChunk();
        while (size > 0) {
            assert(chunk != nullptr);
            const size_type toCopy = std::min(size, chunk->consumed);
            append(chunk->data, toCopy);
            size -= toCopy;
            chunk = chunk->next;
        }
        return *this;
    }

    template<typename T>
    basic_stringbuilder& append(const T& v)
    {
        sb_appender<basic_stringbuilder, T>{}(*this, v);
        return *this;
    }

    template<typename AnyT>
    basic_stringbuilder& operator<<(AnyT&& any)
    {
        return append(std::forward<AnyT>(any));
    }

    template<typename AnyT>
    basic_stringbuilder& append_many(AnyT&& any)
    {
        return append(std::forward<AnyT>(any));
    }

    template<typename AnyT1, typename... AnyTX>
    basic_stringbuilder& append_many(AnyT1&& any1, AnyTX&&... anyX)
    {
        return append(std::forward<AnyT1>(any1)).append_many(std::forward<AnyTX>(anyX)...);
    }

    auto str() const
    {
        const auto size0 = size();
        auto str = std::basic_string<char_type>{};
        str.reserve(size0);
        for (const Chunk* chunk = headChunk(); chunk != nullptr; chunk = chunk->next)
        {
            str.append(chunk->data, chunk->consumed);
        }
        return str;
    }

    bool is_linear() const
    {
        const auto* chunk = headChunk();
        bool has_data = chunk->consumed > 0;
        while (chunk->next) {
            chunk = chunk->next;
            if (chunk->consumed > 0) {
                if (has_data)
                    return false;
                has_data = true;
            }
        }
        return true;
    }

    auto str_view() const
    {
        assert(is_linear());
        const auto* chunk = headChunk();
        while (chunk->consumed == 0) {
            chunk = chunk->next;
            assert(chunk != nullptr);
        }
        return std::basic_string_view<char_type>{ chunk->data, chunk->consumed };
    }

private:
    Chunk* headChunk() noexcept { return reinterpret_cast<Chunk*>(&headChunkInPlace); }
    const Chunk* headChunk() const noexcept { return reinterpret_cast<const Chunk*>(&headChunkInPlace); }

    char_type* claim(size_type exact)
    {
        if (unlikely(tailChunk->reserved - tailChunk->consumed < exact))
            prepareSpace(exact);

        char_type* const claimedChars = &tailChunk->data[tailChunk->consumed];
        tailChunk->consumed += exact;
        return claimedChars;
    }

    std::pair<char_type*, size_type> claim(size_type minimum, size_type maximum)
    {
        assert(maximum >= minimum);
        assert(tailChunk->reserved >= tailChunk->consumed);

        if (unlikely(tailChunk->reserved - tailChunk->consumed < minimum))
            prepareSpace(minimum, maximum);

        assert(tailChunk->reserved >= tailChunk->consumed);
        assert(minimum <= tailChunk->reserved - tailChunk->consumed);

        const size_type claimedSize = std::min(maximum, tailChunk->reserved - tailChunk->consumed);
        const auto claimed = std::make_pair(&tailChunk->data[tailChunk->consumed], claimedSize);
        tailChunk->consumed += claimedSize;
        return claimed;
    }

    Char& claimOne()
    {
        if (unlikely(tailChunk->reserved - tailChunk->consumed < 1))
            prepareSpace(1);
        return tailChunk->data[tailChunk->consumed++];
    }

    void reclaim(size_t exact)
    {
        assert(tailChunk->consumed >= exact);
        tailChunk->consumed -= exact;
    }

    noinline void prepareSpace(size_type minimum)
    {
        if (tailChunk->next == nullptr) {
            tailChunk->next = allocChunk(minimum);
            tailChunk = tailChunk->next;
        }
        else {
            while (true) {
                tailChunk = tailChunk->next;
                assert(tailChunk->consumed == 0);
                if (tailChunk->reserved >= minimum)
                    return;

                if (tailChunk->next == nullptr)
                    tailChunk->next = allocChunk(minimum);
            }
        }
    }

    noinline void prepareSpace(size_type minimum, size_type maximum)
    {
        if (tailChunk->next == nullptr) {
            tailChunk->next = allocChunk(maximum);
            tailChunk = tailChunk->next;
        }
        else
        {
            while (true) {
                tailChunk = tailChunk->next;
                assert(tailChunk->consumed == 0);

                if (tailChunk->reserved >= minimum)
                    return;

                if (tailChunk->next == nullptr)
                    tailChunk->next = allocChunk(maximum);
            }
        }
    }

    size_type determineNextChunkSize(size_type minimum) const noexcept { return std::max(2 * tailChunk->reserved, minimum); }

    constexpr static size_type roundToL1DataCacheLine(size_type size) noexcept
    {
        constexpr size_type l1DataCacheLineSize = 64; //std::hardware_destructive_interference_size;
        return ((l1DataCacheLineSize - 1) + size) / l1DataCacheLineSize * l1DataCacheLineSize;
    }

    Chunk* allocChunk(size_type minimum)
    {
        assert(minimum > 0);
        const auto chunkTotalSize = roundToL1DataCacheLine(determineNextChunkSize(minimum) + sizeof(ChunkHeader));
        auto* rawChunk = AllocTraits::allocate(AllocProvider::get_rebound_allocator(), chunkTotalSize, tailChunk);
        auto* chunk = reinterpret_cast<Chunk*>(rawChunk);
        AllocTraits::construct(AllocProvider::get_rebound_allocator(), chunk, chunkTotalSize - sizeof(ChunkHeader));
        return chunk;
    }

private:
    ChunkInPlace<InPlaceSize> headChunkInPlace;
    Chunk* tailChunk = headChunk();
};


namespace std
{
    template<typename CharT, size_t InPlaceSize, bool Forward, typename Traits>
    inline auto to_string(const basic_inplace_stringbuilder<CharT, InPlaceSize, Forward, Traits>& sb)
    {
        return sb.str();
    }

    template<typename CharT, size_t InPlaceSize, typename Traits, typename Alloc>
    inline auto to_string(const basic_stringbuilder<CharT, InPlaceSize, Traits, Alloc>& sb)
    {
        return sb.str();
    }
}

template<int MaxSize, bool Forward = true, typename Traits = std::char_traits<char>>
using inplace_stringbuilder = basic_inplace_stringbuilder<char, MaxSize, Forward, Traits>;

template<int MaxSize, bool Forward = true, typename Traits = std::char_traits<wchar_t>>
using inplace_wstringbuilder = basic_inplace_stringbuilder<wchar_t, MaxSize, Forward, Traits>;

template<int MaxSize, bool Forward = true, typename Traits = std::char_traits<char16_t>>
using inplace_u16stringbuilder = basic_inplace_stringbuilder<char16_t, MaxSize, Forward, Traits>;

template<int MaxSize, bool Forward = true, typename Traits = std::char_traits<char32_t>>
using inplace_u32stringbuilder = basic_inplace_stringbuilder<char32_t, MaxSize, Forward, Traits>;


template<int InPlaceSize = 0, typename Traits = std::char_traits<char>,typename Alloc = std::allocator<char>>
using stringbuilder = basic_stringbuilder<char, InPlaceSize, Traits, Alloc>;

template<int InPlaceSize = 0, typename Traits = std::char_traits<wchar_t>, typename Alloc = std::allocator<wchar_t>>
using wstringbuilder = basic_stringbuilder<wchar_t, InPlaceSize, Traits, Alloc>;

template<int InPlaceSize = 0, typename Traits = std::char_traits<char16_t>, typename Alloc = std::allocator<char16_t>>
using u16stringbuilder = basic_stringbuilder<char16_t, InPlaceSize, Traits, Alloc>;

template<int InPlaceSize = 0, typename Traits = std::char_traits<char32_t>, typename Alloc = std::allocator<char32_t>>
using u32wstringbuilder = basic_stringbuilder<char32_t, InPlaceSize, Traits, Alloc>;


template<typename SB, typename IntegerT>
struct sb_appender<SB, IntegerT, std::enable_if_t<
    std::is_integral<IntegerT>::value && !std::is_same<IntegerT, typename SB::char_type>::value >>
{
    void operator()(SB& sb, IntegerT iv) const
    {
        // In this particular case, std::div() is x2 slower instead of / and %.
        if (iv >= 0) {
            if (iv >= 10) {
                basic_inplace_stringbuilder<typename SB::char_type, 20, false, typename SB::traits_type> bss;
                do {
                    bss.append(static_cast<typename SB::char_type>('0' + iv % 10));
                    iv /= 10;
                } while (iv > 0);
                sb.append(bss);
            } else {
                sb.append(static_cast<typename SB::char_type>('0' + static_cast<char>(iv)));
            }
        } else {
            if (iv <= -10) {
                basic_inplace_stringbuilder<typename SB::char_type, 20, false, typename SB::traits_type> bss;
                do {
                    bss.append(static_cast<typename SB::char_type>('0' - iv % 10));
                    iv /= 10;
                } while (iv < 0);
                bss.append('-');
                sb.append(bss);
            } else {
                sb.append('-');
                sb.append(static_cast<typename SB::char_type>('0' - static_cast<char>(iv)));
            }
        }
    }
};


template<typename CharT, size_t N, size_t... IX>
struct constexpr_str
{
private:
    const std::array<CharT, N> arr;
    const CharT c_str_[N];

public:
    constexpr constexpr_str(const std::array<CharT, N> arr_, const std::index_sequence<IX...>) :
        arr{arr_},
        c_str_{ arr_[IX]... }
    { }

    constexpr size_t size() const { return N - 1; }
    constexpr const CharT* c_str() const { return c_str_; }
    constexpr auto str() const { return std::string(c_str_, size()); }
};


namespace detail
{
    template <typename T>
    struct type { using value_type = T; };

    template<typename CharT>
    constexpr int estimateTypeSize(type<CharT>) {
        return 1;
    }

    template<typename CharT, typename IntegralT>
    constexpr int estimateTypeSize(type<IntegralT>, std::enable_if_t<std::is_integral<IntegralT>::value && !std::is_same<CharT, IntegralT>::value>* = 0) {
        return 20;
    }

    template<typename CharT, size_t StrSizeWith0>
    constexpr int estimateTypeSize(type<const CharT[StrSizeWith0]>) {
        return StrSizeWith0 - 1;
    }

    template<typename CharT, size_t ExpectedSize, typename StringT>
    constexpr int estimateTypeSize(type<sized_str_t<ExpectedSize, StringT>>) {
        return ExpectedSize;
    }

    template<typename CharT, typename T>
    constexpr int estimateTypeSeqSize(type<T> t) {
        return estimateTypeSize<CharT>(t);
    }

    template<typename CharT, typename T1, typename... TX>
    constexpr int estimateTypeSeqSize(type<T1> t1, type<TX>... tx) {
        return estimateTypeSize<CharT>(t1) + estimateTypeSeqSize<CharT>(tx...);
    }


    template<size_t S1, size_t S2, std::size_t... I1, std::size_t... I2>
    constexpr auto concatenateArrayPair(const std::array<char, S1> arr1, const std::array<char, S2> arr2, std::index_sequence<I1...>, std::index_sequence<I2...>)
    {
        return std::array<char, S1 + S2>{ arr1[I1]..., arr2[I2]... };
    }

    template<size_t S>
    constexpr auto concatenateArrays(const std::array<char, S> arr)
    {
        return arr;
    }

    template<size_t S1, size_t S2, size_t... SX>
    constexpr auto concatenateArrays(const std::array<char, S1> arr1, const std::array<char, S2> arr2, const std::array<char, SX>... arrX)
    {
        return concatenateArrays(concatenateArrayPair(arr1, arr2, std::make_index_sequence<arr1.size()>(), std::make_index_sequence<arr2.size()>()), arrX...);
    }

    template<size_t S1, size_t S2>
    constexpr auto concatenateArrays(const std::array<char, S1> arr1, const std::array<char, S2> arr2)
    {
        return concatenateArrayPair(arr1, arr2, std::make_index_sequence<arr1.size()>(), std::make_index_sequence<arr2.size()>());
    }


    template<typename CharT>
    constexpr std::array<char, 1> stringify(CharT c) {
        return { c };
    }

    template<typename CharT, typename IntegralT>
    constexpr std::enable_if_t<std::is_integral<IntegralT>::value && !std::is_same<CharT, IntegralT>::value> stringify(IntegralT) = delete;

    template<typename CharT, size_t N, size_t... IX>
    constexpr std::array<CharT, sizeof...(IX)> stringify(const CharT(&c)[N], std::index_sequence<IX...>) {
        return { c[IX]... };
    }

    template<typename CharT, size_t N>
    constexpr std::array<char, N - 1> stringify(const CharT(&c)[N]) {
        return stringify<CharT>(c, std::make_index_sequence<N - 1>());
    }


    template<typename CharT, typename T, typename = std::void_t<>>
    struct CanStringify : std::false_type {};

    template<typename CharT, typename T>
    struct CanStringify<CharT, T, std::void_t<decltype(stringify<CharT>(std::declval<T>()))>> : std::true_type {};

    template<typename CharT, typename T>
    constexpr bool canStringify(type<T>) {
        return CanStringify<CharT, T>::value;
    }

    template<typename CharT, typename T1, typename... TX>
    constexpr bool canStringify(type<T1>, type<TX>... tx) {
        return CanStringify<CharT, T1>::value && canStringify<CharT>(tx...);
    }


    template<typename CharT, bool StringifyConstexpr>
    struct StringMaker
    {
        template<typename... TX>
        auto operator()(TX&&... vx) const
        {
            constexpr size_t estimatedSize = estimateTypeSeqSize<CharT>(type<std::remove_reference_t<std::remove_cv_t<TX>>>{}...);
            basic_stringbuilder<CharT, estimatedSize, std::char_traits<CharT>, std::allocator<uint8_t>> sb;
            sb.append_many(std::forward<TX>(vx)...);
            return sb.str();
        }
    };

    template<typename CharT>
    struct StringMaker<CharT, true>
    {
        template<size_t N, size_t... IX>
        constexpr auto data_of(const std::array<CharT, N> arr, const std::index_sequence<IX...> ix) const
        {
            return constexpr_str<CharT, N, IX...>{arr, ix};
        }

        template<size_t N>
        constexpr auto data_of(const std::array<CharT, N> arr) const
        {
            return data_of(arr, std::make_index_sequence<N>());
        }

        template<typename... TX>
        constexpr auto operator()(TX&&... vx) const
        {
            return data_of(concatenateArrays(stringify<CharT>(vx)..., stringify<CharT>('\0')));
        }
    };
}

template<typename... TX>
constexpr auto make_stringbuilder(TX&&... vx)
{
    constexpr size_t estimatedSize = detail::estimateTypeSeqSize<char>(detail::type<TX>{}...);
    stringbuilder<estimatedSize> sb;
    sb.append_many(std::forward<TX>(vx)...);
    return sb;
}

template<typename... TX>
constexpr auto make_string(TX&&... vx)
{
    constexpr bool stringifyConstexpr = detail::canStringify<char>(detail::type<TX>{}...);
    return detail::StringMaker<char, stringifyConstexpr>{}(std::forward<TX>(vx)...);
}
