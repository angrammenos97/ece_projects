`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   23:26:07 01/12/2022
// Design Name:   Proccessor
// Module Name:   /home/ise/verilog_files/Processor_Design/processor_tb.v
// Project Name:  Processor_Design
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: Proccessor
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module processor_tb;

	// Inputs
	reg Clk;
	reg Reset;

	// Outputs
	wire [31:0] RF_A_buf;
	wire [31:0] RF_B_buf;

	// Instantiate the Unit Under Test (UUT)
	Proccessor uut (
		.RF_A_buf(RF_A_buf), 
		.RF_B_buf(RF_B_buf), 
		.Clk(Clk), 
		.Reset(Reset)
	);

	initial begin
		// *********************************************signal initialization********************************************
		// initialize clock
		Clk = 0; // start clock from 1 to allow for reset signal to take effect                                    
		
		// Reset state and PC
		Reset      = 1;
		#3 Reset = 0;
		//#0.5 Reset = 0;

		end
        
		// Add stimulus here
      // clock's period is 4ns, since the polarity changes every 2ns (half period)
		always 
			begin
				#2 Clk = ~Clk;
			end
		
      
endmodule

