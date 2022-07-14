#pragma once

#include <map>
#include <string>
#include "nlohmann/json.hpp"

/************************************************************************/
/* Comparator for case-insensitive comparison in STL assos. containers  */
/************************************************************************/
struct ci_less {
	// case-independent (ci) compare_less binary function
	struct nocase_compare {
		bool operator() (const unsigned char& c1, const unsigned char& c2) const;
	};
	bool operator()(const std::string& s1, const std::string& s2) const;
};

template<class Key, class T, class IgnoredLess = std::less<Key>,
		class Allocator = std::allocator<std::pair<const Key, T>>>
struct Case_Insensitive_Map : std::map<Key, T, struct ci_less> {
	using key_type = Key;
	using mapped_type = T;
};

using json = nlohmann::basic_json<Case_Insensitive_Map>;

bool nocase_equal(const unsigned char& c1, const unsigned char& c2);

bool ci_str_equal(const std::string& s1, const std::string& s2);
