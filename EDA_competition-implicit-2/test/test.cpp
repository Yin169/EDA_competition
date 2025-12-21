#include "convert.hpp"
#include "alphawrap.hpp"
#include "OBJToBNDConverter.hpp"

int main(){
	
	Convert("./data/Silicon_etch_result.bnd");
	// Convert("./remote/Silicon_etch.bnd");
	// Convert("./test/test3.bnd");
    Wrapper("./initial_struct.obj", 600, 600);

    std::string obj_file = "initial_struct_600_600.obj";
    // std::string obj_file = "../remote/Silicon_etch.obj";
    std::string bnd_file = "Silicon_etch_result_end.bnd";
    
    // if (ConvertOBJToDFISE(obj_file, bnd_file)) {
    //     std::cerr << "Conversion failed!" << std::endl;
    //     return 1;
    // }

	return 1;
}