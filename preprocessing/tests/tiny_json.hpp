// Tiny read-only JSON parser for cross-check tests. Handles exactly what
// gen_reference.py produces: objects, arrays, strings, numbers, booleans.
// Not a general-purpose library. Deliberately kept ~200 lines so we can
// review it by eye.
#pragma once

#include <cctype>
#include <cstddef>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace tinyjson {

struct Value;
using Array  = std::vector<Value>;
using Object = std::unordered_map<std::string, Value>;

struct Value {
    enum class Kind { Null, Bool, Number, String, Array, Object };
    Kind kind = Kind::Null;
    double      num  = 0.0;
    bool        bol  = false;
    std::string str;
    Array       arr;
    Object      obj;

    bool        is_obj()    const { return kind == Kind::Object; }
    bool        is_arr()    const { return kind == Kind::Array;  }
    bool        is_num()    const { return kind == Kind::Number; }
    bool        is_str()    const { return kind == Kind::String; }
    double      as_num()    const { return num; }
    const std::string& as_str() const { return str; }
    const Array&  as_arr()  const { return arr; }
    const Object& as_obj()  const { return obj; }
    const Value& operator[](const std::string& k) const {
        auto it = obj.find(k);
        if (it == obj.end()) throw std::runtime_error("tinyjson: missing key '" + k + "'");
        return it->second;
    }
    const Value& operator[](size_t i) const {
        if (i >= arr.size()) throw std::runtime_error("tinyjson: array index out of range");
        return arr[i];
    }
};

class Parser {
public:
    explicit Parser(const std::string& s) : s_(s), i_(0) {}
    Value parse() { skip_ws(); Value v = parse_value(); skip_ws();
        if (i_ != s_.size()) throw std::runtime_error("tinyjson: trailing garbage");
        return v; }
private:
    const std::string& s_; size_t i_;
    void skip_ws() { while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) ++i_; }
    char peek() { if (i_ >= s_.size()) throw std::runtime_error("tinyjson: EOF"); return s_[i_]; }
    char take() { if (i_ >= s_.size()) throw std::runtime_error("tinyjson: EOF"); return s_[i_++]; }
    void expect(char c) { if (take() != c) throw std::runtime_error(std::string("tinyjson: expected '") + c + "'"); }

    Value parse_value() {
        skip_ws();
        char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return parse_string();
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        return parse_number();
    }
    Value parse_object() {
        Value v; v.kind = Value::Kind::Object;
        expect('{'); skip_ws();
        if (peek() == '}') { take(); return v; }
        while (true) {
            skip_ws();
            Value k = parse_string();
            skip_ws(); expect(':');
            Value x = parse_value();
            v.obj.emplace(k.str, std::move(x));
            skip_ws();
            char c = take();
            if (c == ',') continue;
            if (c == '}') break;
            throw std::runtime_error("tinyjson: expected , or } in object");
        }
        return v;
    }
    Value parse_array() {
        Value v; v.kind = Value::Kind::Array;
        expect('['); skip_ws();
        if (peek() == ']') { take(); return v; }
        while (true) {
            v.arr.push_back(parse_value());
            skip_ws();
            char c = take();
            if (c == ',') continue;
            if (c == ']') break;
            throw std::runtime_error("tinyjson: expected , or ] in array");
        }
        return v;
    }
    Value parse_string() {
        Value v; v.kind = Value::Kind::String;
        expect('"');
        std::string out;
        while (true) {
            if (i_ >= s_.size()) throw std::runtime_error("tinyjson: EOF in string");
            char c = s_[i_++];
            if (c == '"') break;
            if (c == '\\') {
                if (i_ >= s_.size()) throw std::runtime_error("tinyjson: EOF in escape");
                char e = s_[i_++];
                switch (e) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    case 'n': out += '\n'; break;
                    case 'r': out += '\r'; break;
                    case 't': out += '\t'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    default:  out += e; break;  // skip unicode handling — not used here
                }
            } else out += c;
        }
        v.str = std::move(out);
        return v;
    }
    Value parse_bool() {
        Value v; v.kind = Value::Kind::Bool;
        if (s_.compare(i_, 4, "true")  == 0) { i_ += 4; v.bol = true;  return v; }
        if (s_.compare(i_, 5, "false") == 0) { i_ += 5; v.bol = false; return v; }
        throw std::runtime_error("tinyjson: bad bool");
    }
    Value parse_null() {
        if (s_.compare(i_, 4, "null") == 0) { i_ += 4; return Value{}; }
        throw std::runtime_error("tinyjson: bad null");
    }
    Value parse_number() {
        size_t start = i_;
        if (s_[i_] == '-' || s_[i_] == '+') ++i_;
        while (i_ < s_.size() && (std::isdigit(static_cast<unsigned char>(s_[i_])) ||
               s_[i_] == '.' || s_[i_] == 'e' || s_[i_] == 'E' || s_[i_] == '+' || s_[i_] == '-'))
            ++i_;
        Value v; v.kind = Value::Kind::Number;
        v.num = std::stod(s_.substr(start, i_ - start));
        return v;
    }
};

inline Value parse_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("tinyjson: cannot open '" + path + "'");
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    Parser p(s);
    return p.parse();
}

}  // namespace tinyjson
