// verify.h — claim checking for the BBQ Experiment & Analysis artifact.
//
// Every number the paper states about its experiments is one row of exp/claims.csv:
//
//   id, section, claim, file, select, column, expected, tolerance
//
//   file       result CSV, relative to exp/
//   select     unique row selector: "key=value;key=value" (exact string match on CSV cells)
//   column     the CSV column holding the measured value
//   expected   the value as printed in the paper
//   tolerance  absolute tolerance (one unit of the last printed digit)
//
// A claim PASSes when exactly one row matches the selector and |measured - expected| <= tolerance.
// Anything else (missing file, 0 or >1 matching rows, non-numeric cell, out of tolerance) FAILs,
// with the reason reported. Header-only, C++17, no dependencies.
#pragma once

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
    std::string id, section, text, file, select, column;
    double expected = 0, tolerance = 0;
};

struct Outcome {
    bool pass = false;
    double measured = NAN;
    std::string reason;  // empty on PASS
};

class Checker {
public:
    explicit Checker(std::string root) : root_(std::move(root)) {}

    Outcome check(const Claim& c) {
        Outcome o;
        const Table* t = table(c.file, o.reason);
        if (!t) return o;
        auto sel = parse_selector(c.select);
        const Row* hit = nullptr;
        int n = 0;
        for (auto& r : t->rows)
            if (matches(r, sel)) { hit = &r; ++n; }
        if (n != 1) { o.reason = "selector matched " + std::to_string(n) + " rows"; return o; }
        auto it = hit->find(c.column);
        if (it == hit->end()) { o.reason = "no column " + c.column; return o; }
        try { o.measured = std::stod(it->second); }
        catch (...) { o.reason = "non-numeric cell '" + it->second + "'"; return o; }
        double diff = std::fabs(o.measured - c.expected);
        o.pass = diff <= c.tolerance * (1 + 1e-9);
        if (!o.pass) {
            char buf[96];
            std::snprintf(buf, sizeof buf, "off by %.6g (tol %.6g)", diff, c.tolerance);
            o.reason = buf;
        }
        return o;
    }

private:
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
        c.select = r["select"]; c.column = r["column"];
        try { c.expected = std::stod(r["expected"]); c.tolerance = std::stod(r["tolerance"]); }
        catch (...) { err = "bad expected/tolerance in claim " + c.id; return false; }
        out.push_back(std::move(c));
    }
    return true;
}

}  // namespace bbq::verify
