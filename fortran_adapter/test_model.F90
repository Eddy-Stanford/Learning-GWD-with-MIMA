#define errcheck if(ierror/=0) then;call err_print;stop;endif
program test_model
  use forpy_mod
  implicit none

  integer :: ierror
  real :: return_prediction
  type(tuple) :: args
  type(module_py) :: model
  type(object) :: return_value
  type(list) :: paths
  ! character(len=:), allocatable :: return_string

  integer, parameter :: NROWS = 1
  integer, parameter :: NCOLS = 486
  integer :: jj
  type(ndarray) :: arr

  ! Create arr = [1,2,3,4,5,6,7,8,9,10]
  real :: matrix(NROWS, NCOLS)

  ierror = forpy_initialize()

  ! Instead of setting the environment variable PYTHONPATH,
  ! we can add the current directory "." to sys.path
  ierror = get_sys_path(paths)
  ierror = paths%append(".")
  
  ierror = import_py(model, "model_api")
  
  do jj = 1, NCOLS
      matrix(jj, 1) = 1
  enddo

  ! Create numpy array
  ierror = ndarray_create(arr, matrix)

  ! Python: 
  ! return_value = mymodule.make_model(12, "Hi", True, message="Hello world!")
  ierror = tuple_create(args, 2)
  ierror = args%setitem(0, arr)
  ierror = args%setitem(1, .true.)
  
  ierror = call_py(return_value, model, "predict", args)
  errcheck

  ierror = cast_nonstrict(return_prediction, return_value)
  errcheck

  write(*,*) "Prediction is ", return_prediction

  ! For call_py, args and kwargs are optional
  ! use call_py_noret to ignore the return value
  ! E. g.:
  ! ierror = call_py_noret(mymodule, "print_args")

  call args%destroy
  call arr%destroy
  call model%destroy
  call return_value%destroy
  call paths%destroy
  
  call forpy_finalize

end program
