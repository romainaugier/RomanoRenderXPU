#pragma once

#if !defined(__ROMANORENDER_PROPERTY)
#define __ROMANORENDER_PROPERTY

#include "romanorender/romanorender.h"

#include <memory>
#include <stdexcept>
#include <vector>

ROMANORENDER_NAMESPACE_BEGIN

template <typename T>
class Property
{
private:
    T* _data = nullptr;
    bool _owned = false;

public:
    Property() = default;

    explicit Property(const T& data) : _data(new T(data)), _owned(true) {}

    Property(const T* ref) : _data(ref), _owned(false) {}

    ~Property()
    {
        if(this->_owned && this->_data != nullptr)
        {
            delete this->_data;
        }
    }

    Property(const Property& other)
    {
        if(other._owned)
        {
            this->_data = new T(*other._data);
            this->_owned = true;
        }
        else
        {
            this->_data = other._data;
            this->_owned = false;
        }
    }

    Property(Property&& other) noexcept : _data(other._data), _owned(other._owned)
    {
        other._data = nullptr;
        other._owned = false;
    }

    Property& operator=(const Property& other)
    {
        if(this != &other)
        {
            if(this->_owned && this->_data != nullptr)
            {
                delete this->_data;
            }

            this->_data = nullptr;
            this->_owned = false;

            if(other._owned)
            {
                this->_data = new T(*other._data);
                this->_owned = true;
            }
            else
            {
                this->_data = other->_data;
                this->_owned = false;
            }
        }

        return *this;
    }

    Property& operator=(Property&& other) noexcept
    {
        if(this != &other)
        {
            if(this->_owned && this->_data != nullptr)
            {
                delete this->_data;
            }

            this->_data = other._data;
            this->_owned = other._owned;
            other._data = nullptr;
            other._owned = false;
        }

        return *this;
    }

    const T& get() const
    {
        ROMANORENDER_ASSERT(this->_data != nullptr, "Trying to get null data");
        return *this->_data;
    }

    T& get()
    {
        ROMANORENDER_ASSERT(this->_data != nullptr, "Trying to get null data");
        return *this->_data;
    }

    const T* get_ptr() const { return this->_data; }

    T* get_ptr() { return this->_data; }

    void set(const T& data)
    {
        if(this->_owned && this->_data != nullptr)
        {
            delete this->_data;
        }

        this->_data = new T(data);
        this->_owned = true;
    }

    void set(T&& data)
    {
        if(this->_owned && this->_data != nullptr)
        {
            delete this->_data;
        }

        this->_data = new T(std::move(data));
        this->_owned = true;
    }

    void reference(const T* ref)
    {
        if(this->_owned && this->_data != nullptr)
        {
            delete this->_data;
        }

        this->_data = const_cast<T*>(ref);
        this->_owned = false;
    }

    bool owns_data() const { return this->_owned; }

    bool initialized() const { return this->_data != nullptr; }
};

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_PROPERTY) */