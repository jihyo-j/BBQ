// Check every paper claim in exp/claims.csv against the stored result CSVs.
//
//   make verify                         # build + run, from the repo root
//   ./build/verify [--claims exp/claims.csv] [--root exp] [--only F12.] [--quiet]
//
// Exit status: 0 if every claim passes, 1 otherwise.
#include <cstring>
#include <iostream>
#include <string>

#include "common/verify.h"

int main(int argc, char** argv) {
    std::string root = "exp", claims_path, only;
    bool quiet = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--root" && i + 1 < argc) root = argv[++i];
        else if (a == "--claims" && i + 1 < argc) claims_path = argv[++i];
        else if (a == "--only" && i + 1 < argc) only = argv[++i];
        else if (a == "--quiet") quiet = true;
        else { std::cerr << "usage: verify [--root DIR] [--claims CSV] [--only ID_PREFIX] [--quiet]\n"; return 2; }
    }
    if (claims_path.empty()) claims_path = root + "/claims.csv";

    std::vector<bbq::verify::Claim> claims;
    std::string err;
    if (!bbq::verify::load_claims(claims_path, claims, err)) { std::cerr << err << "\n"; return 2; }

    bbq::verify::Checker checker(root);
    int pass = 0, fail = 0;
    for (auto& c : claims) {
        if (!only.empty() && c.id.rfind(only, 0) != 0) continue;
        auto o = checker.check(c);
        o.pass ? ++pass : ++fail;
        if (quiet && o.pass) continue;
        std::printf("%-4s %-7s %-9s %-58.58s paper=%-9g measured=%-12.6g %s\n", o.pass ? "PASS" : "FAIL",
                    c.id.c_str(), c.section.c_str(), c.text.c_str(), c.expected, o.measured, o.reason.c_str());
    }
    std::printf("\n%d/%d claims verified%s\n", pass, pass + fail, fail ? "" : " — all paper numbers reproduce from results/");
    return fail ? 1 : 0;
}
