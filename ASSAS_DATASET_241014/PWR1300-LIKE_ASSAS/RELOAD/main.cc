#include <iostream>
#include "astec_interface.hpp"
#include "odessa.hpp"

int main( void )
{
    /* A classical compuation
    astec::init( "reload.mdat" );
    int icont ;
    icont = astec::steady() ;
    while( icont )
    {
      astec::calc();
      icont =  astec::tool();
    }
    end();
    */

    std::cout << "Start test..." << std::endl ;

    // Initialize computation
    std::cout << "Initialize computation" << std::endl ;
    astec::init( "reload.mdat" );
    int icont ;
    icont = astec::steady() ;

    // Make 10 time steps
    for( int step = 0 ; step < 10 ; ++step )
    {
      astec::calc();
      icont =  astec::tool();
    }

    // After the following time step, simulate an event to ask a saving
    std::cout << "A time step before asking the saving" << std::endl ;
    astec::calc();
    std::cout << "Simulate an event to ask a saving" << std::endl ;
    odbase root = astec::root_database() ;
    odbase sensor = odsearch( root, "SENSOR", "AskedSav" ) ;
    odessa::odput( sensor, "value", 1. ) ;
    std::cout << "Call to tool will generate the saving" << std::endl ;
    icont =  astec::tool();

    // Make 10 time steps
    for( int step = 0 ; step < 10 ; ++step )
    {
      astec::calc();
      icont =  astec::tool();
    }

    // Create the reload structure
    std::cout << "Create the reload structure" << std::endl ;
    root = astec::root_database() ;
    odbase reload ;
    odessa::odinit( reload ) ;
    odessa::odput( reload, "FILE", "mydir.bin" ) ;
    odessa::odput( reload, "FORM", "DIRZIP" ) ;
    odessa::odput( reload, "TIME", 0.7 ) ;
    odessa::odput( root, "RELOAD", reload ) ;

    // Make one time step which will reload the computation
    std::cout << "Make one time step which will reload the computation" << std::endl ;
    astec::calc();
    icont =  astec::tool();

    // Make 2 additional time steps
    std::cout << "Make 2 additional time steps" << std::endl ;
    for( int step = 0 ; step < 2 ; ++step )
    {
      astec::calc();
      icont =  astec::tool();
    }

    //End the test. Note no pause is done, since the pause occurs during the last call to tool() only if final time is reached
    std::cout << "End the test" << std::endl ;
    astec::end() ;
}
