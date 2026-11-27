// verify.h — claim checking for the BBQ Experiment & Analysis artifact.
//
// Every number the paper states about its experiments is one row of exp/claims.csv:
//
//   id, section, claim, file, select, column, derive, expected, tolerance
//
//   file       result CSV, relative to exp/
//   select     unique row selector: "key=value;key=value" (exact string match on CSV cells)
//   column     the CSV column holding the measured value
//   agg        optional: min | max | mean | median | count over ALL rows the selector matches (for range
//              claims such as "26.0-40.5 %"); empty means the selector must match exactly one row
//   derive     instead of file/select/column: an arithmetic expression over *measured* values of other
//              claims, e.g. "F15.10/F15.9" (ratios, "x faster", "% fewer"). Operators + - * / and ( ).
//   expected   the value as printed in the paper; empty for a *reference* row (an operating point the
//              paper only uses inside a derived number, e.g. the denominator of "2.01x faster")
//   tolerance  absolute tolerance (one unit of the last printed digit)
//
// A direct claim PASSes when exactly one row matches the selector and |measured - expected| <= tolerance.
// A derived claim PASSes when every referenced claim could be measured and the expression is within
// tolerance; it is computed from measured values, never from the paper's printed ones.
// Anything else (missing file, 0 or >1 matching rows, non-numeric cell, out of tolerance) FAILs,
// with the reason reported. Header-only, C++17, no dependencies.
#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace bbq::verify {

using Row = std::map<std::string, std::string>;

// ---------------------------------------------------------------- CSV (RFC 4180 quoting)
inline std::vector<std::string> split_csv_line(std::istream& in, bool& ok) {
    std::vector<std::string> cells;
    std::string cell;
    bool quoted = false, any = false;
    char c;
    while (in.get(c)) {
        any = true;
        if (quoted) {
            if (c == '"') {
                if (in.peek() == '"') { cell += '"'; in.get(); }
                else quoted = false;
            } else cell += c;
        } else if (c == '"') quoted = true;
        else if (c == ',') { cells.push_back(std::move(cell)); cell.clear(); }
        else if (c == '\n') break;
        else if (c != '\r') cell += c;
    }
    ok = any;
    if (any) cells.push_back(std::move(cell));
    return cells;
}

struct Table {
    std::vector<std::string> header;
    std::vector<Row> rows;
};

inline bool read_csv(const std::string& path, Table& t, std::string& err) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { err = "cannot open " + path; return false; }
    bool ok = false;
    t.header = split_csv_line(in, ok);
    if (!ok) { err = "empty file " + path; return false; }
    while (true) {
        auto cells = split_csv_line(in, ok);
        if (!ok) break;
        if (cells.size() == 1 && cells[0].empty()) continue;
        Row r;
        for (size_t i = 0; i < t.header.size(); ++i) r[t.header[i]] = i < cells.size() ? cells[i] : "";
        t.rows.push_back(std::move(r));
    }
    return true;
}

// ---------------------------------------------------------------- selectors
inline std::vector<std::pair<std::string, std::string>> parse_selector(const std::string& s) {
    std::vector<std::pair<std::string, std::string>> out;
    std::stringstream ss(s);
    std::string term;
    while (std::getline(ss, term, ';')) {
        auto eq = term.find('=');
        if (eq != std::string::npos) out.emplace_back(term.substr(0, eq), term.substr(eq + 1));
    }
    return out;
}

inline bool matches(const Row& r, const std::vector<std::pair<std::string, std::string>>& sel) {
    for (auto& [k, v] : sel) {
        auto it = r.find(k);
        if (it == r.end() || it->second != v) return false;
    }
    return true;
}

// ---------------------------------------------------------------- claims
struct Claim {
    std::string id, section, text, file, select, column, derive, agg;
    double expected = 0, tolerance = 0;
    bool reference = false;  // measured for use in `derive`, not itself a paper claim
};

struct Outcome {
    bool pass = false;
    double measured = NAN;
    std::string reason;  // empty on PASS
};

// Recursive-descent evaluator for `derive` expressions: claim ids, numbers, + - * / ( ).
class Expr {
public:
    Expr(const std::string& s, const std::map<std::string, double>& vals) : s_(s), vals_(vals) {}
    bool eval(double& out, std::string& err) {
        out = sum();
        skip();
        if (err_.empty() && pos_ != s_.size()) err_ = "unexpected '" + s_.substr(pos_, 1) + "'";
        err = err_;
        return err_.empty();
    }

private:
    void skip() { while (pos_ < s_.size() && s_[pos_] == ' ') ++pos_; }
    bool eat(char c) { skip(); if (pos_ < s_.size() && s_[pos_] == c) { ++pos_; return true; } return false; }
    double sum() {
        double v = product();
        for (;;) {
            if (eat('+')) v += product();
            else if (eat('-')) v -= product();
            else return v;
        }
    }
    double product() {
        double v = atom();
        for (;;) {
            if (eat('*')) v *= atom();
            else if (eat('/')) v /= atom();
            else return v;
        }
    }
    double atom() {
        if (eat('(')) { double v = sum(); if (!eat(')')) fail("missing ')'"); return v; }
        if (eat('-')) return -atom();
        skip();
        size_t b = pos_;
        while (pos_ < s_.size() && (std::isalnum(static_cast<unsigned char>(s_[pos_])) || s_[pos_] == '.' || s_[pos_] == '_')) ++pos_;
        std::string tok = s_.substr(b, pos_ - b);
        if (tok.empty()) { fail("expected a value"); return NAN; }
        if (std::isdigit(static_cast<unsigned char>(tok[0]))) return std::stod(tok);
        auto it = vals_.find(tok);
        if (it == vals_.end()) { fail("unmeasured claim " + tok); return NAN; }
        return it->second;
    }
    void fail(const std::string& e) { if (err_.empty()) err_ = e; }
    const std::string& s_;
    const std::map<std::string, double>& vals_;
    size_t pos_ = 0;
    std::string err_;
};

class Checker {
public:
    explicit Checker(std::string root) : root_(std::move(root)) {}

    // Direct claims first, then derived ones (which may reference any direct or earlier derived claim).
    std::vector<Outcome> run(const std::vector<Claim>& claims) {
        std::vector<Outcome> out(claims.size());
        std::map<std::string, double> measured;
        for (int pass = 0; pass < 2; ++pass) {
            for (size_t i = 0; i < claims.size(); ++i) {
                const Claim& c = claims[i];
                if (c.derive.empty() != (pass == 0)) continue;
                Outcome& o = out[i];
                bool ok = pass == 0 ? measure(c, o) : derive(c, measured, o);
                if (!ok) continue;
                measured[c.id] = o.measured;
                judge(c, o);
            }
        }
        return out;
    }

private:
    bool measure(const Claim& c, Outcome& o) {
        const Table* t = table(c.file, o.reason);
        if (!t) return false;
        auto sel = parse_selector(c.select);
        std::vector<const Row*> hits;
        for (auto& r : t->rows)
            if (matches(r, sel)) hits.push_back(&r);
        if (c.agg.empty()) {
            if (hits.size() != 1) { o.reason = "selector matched " + std::to_string(hits.size()) + " rows"; return false; }
        } else if (hits.empty()) {
            o.reason = "selector matched 0 rows"; return false;
        }
        std::vector<double> v;
        for (const Row* h : hits) {
            auto it = h->find(c.column);
            if (it == h->end()) { o.reason = "no column " + c.column; return false; }
            try { v.push_back(std::stod(it->second)); }
            catch (...) { o.reason = "non-numeric cell '" + it->second + "'"; return false; }
        }
        if (c.agg.empty()) { o.measured = v[0]; return true; }
        std::sort(v.begin(), v.end());
        double sum = 0;
        for (double x : v) sum += x;
        if (c.agg == "min") o.measured = v.front();
        else if (c.agg == "max") o.measured = v.back();
        else if (c.agg == "mean") o.measured = sum / v.size();
        else if (c.agg == "count") o.measured = static_cast<double>(v.size());
        else if (c.agg == "median") o.measured = v.size() % 2 ? v[v.size() / 2] : (v[v.size() / 2 - 1] + v[v.size() / 2]) / 2;
        else { o.reason = "unknown agg " + c.agg; return false; }
        return true;
    }
    static bool derive(const Claim& c, const std::map<std::string, double>& measured, Outcome& o) {
        return Expr(c.derive, measured).eval(o.measured, o.reason);
    }
    static void judge(const Claim& c, Outcome& o) {
        if (c.reference) { o.pass = true; return; }
        double diff = std::fabs(o.measured - c.expected);
        o.pass = diff <= c.tolerance * (1 + 1e-9);
        if (!o.pass) {
            char buf[96];
            std::snprintf(buf, sizeof buf, "off by %.6g (tol %.6g)", diff, c.tolerance);
            o.reason = buf;
        }
    }
    const Table* table(const std::string& file, std::string& err) {
        auto it = cache_.find(file);
        if (it != cache_.end()) return &it->second;
        Table t;
        if (!read_csv(root_ + "/" + file, t, err)) return nullptr;
        return &cache_.emplace(file, std::move(t)).first->second;
    }
    std::string root_;
    std::map<std::string, Table> cache_;
};

inline bool load_claims(const std::string& path, std::vector<Claim>& out, std::string& err) {
    Table t;
    if (!read_csv(path, t, err)) return false;
    for (auto& r : t.rows) {
        Claim c;
        c.id = r["id"]; c.section = r["section"]; c.text = r["claim"]; c.file = r["file"];
        c.select = r["select"]; c.column = r["column"]; c.derive = r["derive"]; c.agg = r["agg"];
        c.reference = r["expected"].empty();
        if (!c.reference) try { c.expected = std::stod(r["expected"]); c.tolerance = std::stod(r["tolerance"]); }
        catch (...) { err = "bad expected/tolerance in claim " + c.id; return false; }
        out.push_back(std::move(c));
    }
    return true;
}

}  // namespace bbq::verify
