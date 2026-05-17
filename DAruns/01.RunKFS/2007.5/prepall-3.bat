#in at this stage currents are wrong+Initconc.dat wrong; orther Input files are correct
# output files *.txt, lastconc.xls, and input files Initconc.dat are deleted at the end of this script
                                                                                    
xcopy /Y .\01.hydrol1dep1\BLS 02.hydrol1dep2\BLS

xcopy /Y .\01.hydrol1dep1\BLS 03.hydrol1dep3\BLS   

xcopy /Y .\01.hydrol1dep1\BLS 04.hydrol1dep4\BLS    

xcopy /Y .\01.hydrol1dep1\BLS 05.hydrol2dep1\BLS    

xcopy /Y .\01.hydrol1dep1\BLS 06.hydrol2dep2\BLS     

xcopy /Y .\01.hydrol1dep1\BLS 07.hydrol2dep3\BLS     

xcopy /Y .\01.hydrol1dep1\BLS 08.hydrol2dep4\BLS     

xcopy /Y .\01.hydrol1dep1\BLS 09.hydrol3dep1\BLS     

xcopy /Y .\01.hydrol1dep1\BLS 10.hydrol3dep2\BLS     

xcopy /Y .\01.hydrol1dep1\BLS 11.hydrol3dep3\BLS      

xcopy /Y .\01.hydrol1dep1\BLS 12.hydrol3dep4\BLS     

xcopy /Y .\01.hydrol1dep1\BLS 13.hydrol4dep1\BLS     

xcopy /Y .\01.hydrol1dep1\BLS 14.hydrol4dep2\BLS     

xcopy /Y .\01.hydrol1dep1\BLS 15.hydrol4dep3\BLS     

xcopy /Y .\01.hydrol1dep1\BLS 16.hydrol4dep4\BLS     

xcopy /Y .\01.hydrol1dep1\BLS 17.hydrol5dep1\BLS     

xcopy /Y .\01.hydrol1dep1\BLS 18.hydrol5dep2\BLS      

xcopy /Y .\01.hydrol1dep1\BLS 19.hydrol5dep3\BLS       

xcopy /Y .\01.hydrol1dep1\BLS 20.hydrol5dep4\BLS      

xcopy /Y .\01.hydrol1dep1\BLS 21.hydrol6dep1\BLS      

xcopy /Y .\01.hydrol1dep1\BLS 22.hydrol6dep2\BLS      

xcopy /Y .\01.hydrol1dep1\BLS 23.hydrol6dep3\BLS       

xcopy /Y .\01.hydrol1dep1\BLS 24.hydrol6dep4\BLS     


for /r %%f in (*.txt) do (
    del "%%f"
)





