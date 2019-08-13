 // th_utils.cu ___________________________________________________________________________________________________________________

#include	"thrusters/tenor/th_includes.h"

//_____________________________________________________________________________________________________________________________

struct Th_Utils
{

template <typename Iterator>
	struct ItOut
	{
		typedef typename	std::iterator_traits<Iterator>::value_type	Value;
		typedef typename	std::ostream_iterator< Value>				OIterator; 

		const Iterator		&m_First;
		const Iterator		&m_Last;
		const char			*m_Delim;
		  
		ItOut( const Iterator &first, const Iterator &last, const char *delim)
			: m_First( first), m_Last( last), m_Delim( delim)
		{}

		friend	std::ostream	&operator<<( std::ostream &ostr, const ItOut &ic)
		{
			thrust::copy( ic.m_First, ic.m_Last, std::ostream_iterator< Value>( ostr, ic.m_Delim));  
			return ostr;
		}
	};

template <typename Iterator>
	static auto IterOut( const Iterator &first, const Iterator &last, const  char *delim)
	{
		return Th_Utils::ItOut< Iterator>( first, last, delim);
	}

	//_____________________________________________________________________________________________________________________________ 
};

//_____________________________________________________________________________________________________________________________
