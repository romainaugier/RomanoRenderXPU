#pragma once

#if !defined(__ROMANORENDER_CUDAVECTOR)
#define __ROMANORENDER_CUDAVECTOR

#include "romanorender/cuda_utils.h"
#include "romanorender/romanorender.h"
#include "stdromano/logger.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <functional>

ROMANORENDER_NAMESPACE_BEGIN

template <typename T>
class CudaVector
{
    static constexpr float GROW_RATE = 1.61803398875f;
    static constexpr uint32_t MIN_SIZE = 128;

private:
    T* _data = nullptr;
    uint32_t _capacity = 0;
    uint32_t _size = 0;

    ROMANORENDER_FORCE_INLINE void set_capacity(const uint32_t capacity) noexcept
    {
        this->_capacity = capacity;
    }

    ROMANORENDER_FORCE_INLINE void set_size(const uint32_t size) noexcept { this->_size = size; }

    ROMANORENDER_FORCE_INLINE void incr_size() noexcept { this->_size++; }

    ROMANORENDER_FORCE_INLINE void decr_size() noexcept { this->_size--; }

    static T* allocate(const uint32_t capacity)
    {
        if(capacity == 0)
        {
            return nullptr;
        }

        void* memory = nullptr;
        const size_t size = capacity * sizeof(T);

        CUDA_CHECK(cudaMallocHost(&memory, size, cudaHostAllocDefault));

        return static_cast<T*>(memory);
    }

    static void deallocate(T* data)
    {
        if(data != nullptr)
        {
            CUDA_CHECK(cudaFreeHost(data));
        }
    }

    ROMANORENDER_FORCE_INLINE void grow() noexcept
    {
        if(this->_data == nullptr)
        {
            this->resize(MIN_SIZE);
        }
        else
        {
            this->resize(static_cast<uint32_t>(static_cast<float>(this->capacity()) * GROW_RATE));
        }
    }

public:
    using value_type = T;
    using size_type = uint32_t;
    using difference_type = int32_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = value_type*;
    using const_pointer = const value_type*;

    CudaVector() : _data(nullptr), _capacity(0), _size(0) {}

    CudaVector(const uint32_t initial_capacity)
    {
        const uint32_t capacity = initial_capacity == 0 ? MIN_SIZE : initial_capacity;
        this->_data = CudaVector::allocate(capacity);
        this->set_capacity(capacity);
        this->set_size(0);
    }

    CudaVector(const uint32_t count, const T& value)
    {
        if(count == 0)
        {
            this->_data = CudaVector::allocate(MIN_SIZE);
            this->set_capacity(MIN_SIZE);
            this->set_size(0);
            return;
        }

        this->_data = CudaVector::allocate(count);
        this->set_capacity(count);

        if(this->_data)
        {
            for(uint32_t i = 0; i < count; ++i)
            {
                ::new(this->_data + i) T(value);
            }
            this->set_size(count);
        }
    }

    ~CudaVector()
    {
        if(this->_data != nullptr)
        {
            this->clear();
            CudaVector::deallocate(this->_data);
        }

        this->_data = nullptr;
        this->_capacity = 0;
        this->_size = 0;
    }

    CudaVector(const CudaVector<T>& other)
    {
        if(other._data == nullptr)
        {
            this->_data = nullptr;
            this->_capacity = 0;
            this->_size = 0;
            return;
        }

        this->_data = CudaVector::allocate(other.capacity());
        this->set_capacity(other.capacity());

        if(this->_data)
        {
            for(uint32_t i = 0; i < other.size(); i++)
            {
                ::new(this->data() + i) T(other[i]);
            }
            this->set_size(other.size());
        }
    }

    CudaVector(CudaVector<T>&& other) noexcept
    {
        this->_data = other._data;
        this->_capacity = other._capacity;
        this->_size = other._size;

        other._data = nullptr;
        other._capacity = 0;
        other._size = 0;
    }

    CudaVector& operator=(const CudaVector<T>& other)
    {
        if(this != &other)
        {
            if(this->_data != nullptr)
            {
                this->clear();
                CudaVector::deallocate(this->_data);
            }

            this->_data = nullptr;
            this->_capacity = 0;
            this->_size = 0;

            if(other._data != nullptr)
            {
                this->_data = CudaVector::allocate(other.capacity());
                this->set_capacity(other.capacity());

                if(this->_data)
                {
                    for(uint32_t i = 0; i < other.size(); i++)
                    {
                        ::new(this->data() + i) T(other[i]);
                    }
                    this->set_size(other.size());
                }
            }
        }
        return *this;
    }

    CudaVector& operator=(CudaVector<T>&& other) noexcept
    {
        if(this != &other)
        {
            if(this->_data != nullptr)
            {
                this->clear();
                CudaVector::deallocate(this->_data);
            }

            this->_data = other._data;
            this->_capacity = other._capacity;
            this->_size = other._size;

            other._data = nullptr;
            other._capacity = 0;
            other._size = 0;
        }
        return *this;
    }

    ROMANORENDER_FORCE_INLINE uint32_t capacity() const noexcept { return this->_capacity; }

    ROMANORENDER_FORCE_INLINE uint32_t size() const noexcept { return this->_size; }

    ROMANORENDER_FORCE_INLINE bool empty() const noexcept { return this->size() == 0; }

    ROMANORENDER_FORCE_INLINE void syncToCuda() const noexcept
    {
        cudaMemPrefetchAsync(this->data(), this->size() * sizeof(T), 0);
    }

    ROMANORENDER_FORCE_INLINE void syncFromCuda() const noexcept
    {
        cudaDeviceSynchronize();
        cudaMemPrefetchAsync(this->data(), this->size() * sizeof(T), cudaCpuDeviceId);
    }

    class iterator
    {
        friend class CudaVector;
        CudaVector* vector;
        uint32_t index;

        void advance()
        {
            if(this->index >= this->vector->size())
                this->index = this->vector->size();
        }

    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = T;
        using difference_type = int32_t;
        using pointer = T*;
        using reference = T&;

        iterator(CudaVector* v, uint32_t i) : vector(v), index(i) { this->advance(); }

        iterator& operator++()
        {
            this->index++;
            this->advance();
            return *this;
        }

        bool operator==(const iterator& other) const
        {
            return this->index == other.index && this->vector == other.vector;
        }

        bool operator!=(const iterator& other) const { return !(*this == other); }

        reference operator*() { return (*this->vector)[this->index]; }

        pointer operator->() { return &(*this->vector)[this->index]; }

        iterator& operator--()
        {
            this->index--;
            return *this;
        }

        iterator operator+(difference_type n) const
        {
            return iterator(this->vector, this->index + n);
        }

        iterator operator-(difference_type n) const
        {
            return iterator(this->vector, this->index - n);
        }

        difference_type operator-(const iterator& other) const
        {
            return static_cast<difference_type>(this->index - other.index);
        }

        iterator& operator+=(difference_type n)
        {
            this->index += n;
            this->advance();
            return *this;
        }

        iterator& operator-=(difference_type n)
        {
            this->index -= n;
            this->advance();
            return *this;
        }

        reference operator[](difference_type n) { return (*this->vector)[this->index + n]; }

        bool operator<(const iterator& other) const { return this->index < other.index; }

        bool operator>(const iterator& other) const { return this->index > other.index; }

        bool operator<=(const iterator& other) const { return this->index <= other.index; }

        bool operator>=(const iterator& other) const { return this->index >= other.index; }
    };

    class const_iterator
    {
        friend class CudaVector;
        const CudaVector* vector;
        uint32_t index;

        void advance()
        {
            if(this->index >= this->vector->size())
                this->index = this->vector->size();
        }

    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = T;
        using difference_type = int32_t;
        using pointer = const T*;
        using reference = const T&;

        const_iterator(const CudaVector* v, uint32_t i) : vector(v), index(i) { this->advance(); }

        const_iterator& operator++()
        {
            this->index++;
            this->advance();
            return *this;
        }

        bool operator==(const const_iterator& other) const
        {
            return this->index == other.index && this->vector == other.vector;
        }

        bool operator!=(const const_iterator& other) const { return !(*this == other); }

        reference operator*() const { return (*this->vector)[this->index]; }

        pointer operator->() const { return &(*this->vector)[this->index]; }

        const_iterator& operator--()
        {
            this->index--;
            return *this;
        }

        const_iterator operator+(difference_type n) const
        {
            return const_iterator(this->vector, this->index + n);
        }

        const_iterator operator-(difference_type n) const
        {
            return const_iterator(this->vector, this->index - n);
        }

        difference_type operator-(const const_iterator& other) const
        {
            return static_cast<difference_type>(this->index - other.index);
        }

        const_iterator& operator+=(difference_type n)
        {
            this->index += n;
            this->advance();
            return *this;
        }

        const_iterator& operator-=(difference_type n)
        {
            this->index -= n;
            this->advance();
            return *this;
        }

        reference operator[](difference_type n) const { return (*this->vector)[this->index + n]; }

        bool operator<(const const_iterator& other) const { return this->index < other.index; }

        bool operator>(const const_iterator& other) const { return this->index > other.index; }

        bool operator<=(const const_iterator& other) const { return this->index <= other.index; }

        bool operator>=(const const_iterator& other) const { return this->index >= other.index; }
    };

    iterator begin() { return iterator(this, 0); }

    iterator end() { return iterator(this, this->size()); }

    const_iterator begin() const { return const_iterator(this, 0); }

    const_iterator end() const { return const_iterator(this, this->size()); }

    const_iterator cbegin() const { return const_iterator(this, 0); }

    const_iterator cend() const { return const_iterator(this, this->size()); }

    ROMANORENDER_FORCE_INLINE T* data() noexcept { return this->_data; }

    ROMANORENDER_FORCE_INLINE const T* data() const noexcept { return this->_data; }

    ROMANORENDER_FORCE_INLINE T* at(const uint32_t index) noexcept { return this->data() + index; }

    ROMANORENDER_FORCE_INLINE const T* at(const uint32_t index) const noexcept
    {
        return this->data() + index;
    }

    ROMANORENDER_FORCE_INLINE T& operator[](const uint32_t i) noexcept { return *this->at(i); }

    ROMANORENDER_FORCE_INLINE const T& operator[](const uint32_t i) const noexcept
    {
        return *this->at(i);
    }

    ROMANORENDER_FORCE_INLINE T& front() { return this->operator[](0); }

    ROMANORENDER_FORCE_INLINE const T& front() const { return this->operator[](0); }

    ROMANORENDER_FORCE_INLINE T& back() { return this->operator[](this->size() - 1); }

    ROMANORENDER_FORCE_INLINE const T& back() const { return this->operator[](this->size() - 1); }

    void resize(const uint32_t new_capacity) noexcept
    {
        const uint32_t old_capacity = this->capacity();

        if(new_capacity <= old_capacity)
        {
            return;
        }

        T* new_data = CudaVector<T>::allocate(new_capacity);

        if(new_data == nullptr)
        {
            return;
        }

        if(this->_data != nullptr)
        {
            const uint32_t old_size = this->size();
            const uint32_t new_size = old_size;

            for(uint32_t i = 0; i < new_size; ++i)
            {
                ::new(new_data + i) T(std::move_if_noexcept(this->_data[i]));
                this->_data[i].~T();
            }

            CudaVector<T>::deallocate(this->_data);
        }

        this->_data = new_data;
        this->set_capacity(new_capacity);
    }

    ROMANORENDER_FORCE_INLINE void reserve(const uint32_t new_capacity) noexcept
    {
        this->resize(this->capacity() + new_capacity + 1);
    }

    void push_back(const T& element) noexcept
    {
        if(this->size() == this->capacity())
            this->grow();

        if(this->_data)
        {
            ::new(this->data() + this->size()) T(element);
            this->incr_size();
        }
    }

    void push_back(T&& element) noexcept
    {
        if(this->size() == this->capacity())
            this->grow();

        if(this->_data)
        {
            ::new(this->data() + this->size()) T(std::move(element));
            this->incr_size();
        }
    }

    template <typename... Args>
    T& emplace_back(Args&&... args) noexcept
    {
        if(this->size() == this->capacity())
            this->grow();

        if(this->_data)
        {
            T* ptr = ::new(this->data() + this->size()) T(std::forward<Args&&>(args)...);
            this->incr_size();
            return *ptr;
        }

        static T dummy{};
        return dummy;
    }

    void insert(const T& element, const uint32_t position) noexcept
    {
        if(this->size() == this->capacity())
            this->grow();

        if(this->_data && position <= this->size())
        {
            std::memmove(this->at(position + 1), this->at(position), (this->size() - position) * sizeof(T));
            ::new(this->at(position)) T(element);
            this->incr_size();
        }
    }

    iterator insert(const_iterator pos, uint32_t count, const T& value) noexcept
    {
        if(count == 0 || !this->_data)
            return iterator(this, pos.index);

        const uint32_t position = pos.index;
        if(position > this->size())
            return this->end();

        const uint32_t new_size = this->size() + count;
        while(this->capacity() < new_size)
            this->grow();

        if(this->_data)
        {
            std::memmove(this->at(position + count), this->at(position), (this->size() - position) * sizeof(T));
            for(uint32_t i = 0; i < count; ++i)
            {
                ::new(this->at(position + i)) T(value);
            }
            this->set_size(new_size);
            return iterator(this, position);
        }
        return iterator(this, pos.index);
    }

    template <class InputIt>
    iterator insert(const_iterator pos, InputIt first, InputIt last) noexcept
    {
        if(first == last || !this->_data)
            return iterator(this, pos.index);

        const uint32_t position = pos.index;
        if(position > this->size())
            return this->end();

        const uint32_t count = static_cast<uint32_t>(std::distance(first, last));
        const uint32_t new_size = this->size() + count;
        while(this->capacity() < new_size)
            this->grow();

        if(this->_data)
        {
            std::memmove(this->at(position + count), this->at(position), (this->size() - position) * sizeof(T));
            uint32_t offset = 0;
            for(auto it = first; it != last; ++it, ++offset)
            {
                ::new(this->at(position + offset)) T(*it);
            }
            this->set_size(new_size);
            return iterator(this, position);
        }
        return iterator(this, pos.index);
    }

    iterator erase(const_iterator pos) noexcept
    {
        if(this->_data && pos.index < this->size())
        {
            const uint32_t position = pos.index;
            this->at(position)->~T();
            std::memmove(this->at(position), this->at(position + 1), (this->size() - position - 1) * sizeof(T));
            this->decr_size();
            return iterator(this, position);
        }
        return this->end();
    }

    iterator erase(iterator first, iterator last) noexcept
    {
        if(this->_data && first.index <= this->size() && first.index < last.index && last.index <= this->size())
        {
            const uint32_t start_pos = first.index;
            const uint32_t count = last.index - first.index;

            for(uint32_t i = 0; i < count; ++i)
            {
                this->at(start_pos + i)->~T();
            }

            std::memmove(this->at(start_pos),
                         this->at(start_pos + count),
                         (this->size() - start_pos - count) * sizeof(T));
            this->set_size(this->size() - count);
            return iterator(this, start_pos);
        }
        return this->end();
    }

    void remove(const uint32_t position) noexcept
    {
        if(this->_data && position < this->size())
        {
            this->at(position)->~T();
            std::memmove(this->at(position), this->at(position + 1), (this->size() - position - 1) * sizeof(T));
            this->decr_size();
        }
    }

    T pop_back() noexcept
    {
        if(this->_data && this->size() > 0)
        {
            this->decr_size();
            T object = std::move((*this)[this->size()]);
            (*this)[this->size()].~T();
            return object;
        }
        return T{};
    }

    iterator find(const T& other) noexcept
    {
        if(this->_data)
        {
            for(uint32_t i = 0; i < this->size(); i++)
            {
                if(other == this->operator[](i))
                    return iterator(this, i);
            }
        }
        return this->end();
    }

    const_iterator cfind(const T& other) const noexcept
    {
        if(this->_data)
        {
            for(uint32_t i = 0; i < this->size(); i++)
            {
                if(other == this->operator[](i))
                    return const_iterator(this, i);
            }
        }
        return this->cend();
    }

    ROMANORENDER_FORCE_INLINE void shrink_to_fit() noexcept
    {
        if(this->_capacity > this->_size)
            this->resize(this->size());
    }

    void clear() noexcept
    {
        if(this->_data != nullptr)
        {
            for(uint32_t i = 0; i < this->size(); i++)
            {
                this->operator[](i).~T();
            }
            this->set_size(0);
        }
    }

    template <typename F = std::less<T> >
    ROMANORENDER_FORCE_INLINE void sort(F&& cmp = F()) noexcept
    {
        if(this->_data && this->size() > 1)
        {
            std::sort(this->begin(), this->end(), std::forward<F&&>(cmp));
        }
    }
};

ROMANORENDER_NAMESPACE_END

#endif // !defined(__ROMANORENDER_CUDAVECTOR)