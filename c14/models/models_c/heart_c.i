%module heart_c

%{
    #define SWIG_FILE_WITH_INIT
    #include "heart_c.h"
%}

%include "numpy.i"
 // Get the STL typemaps
%include "stl.i"

// Handle standard exceptions
%include "exception.i"
%exception
{
  try
  {
    $action
  }
  catch (const std::invalid_argument& e)
  {
    SWIG_exception(SWIG_ValueError, e.what());
  }
  catch (const std::out_of_range& e)
  {
    SWIG_exception(SWIG_IndexError, e.what());
  }
}
%init %{
    import_array();
%}

%apply (int* ARGOUT_ARRAY1, int DIM1) {(int* rangevec, int n)}

%apply (int DIM1 , int DIM2 , double* INPLACE_ARRAY2)
      {(int nrows, int ncols, double* indata),(int nrows2, int ncols2, double* ipara)};
%apply (int DIM1 , double* INPLACE_ARRAY1)
      {(int nn, double* new_data),(int no , double* old_data),(int nc, double* Catm)};
%include "heart_c.h"
