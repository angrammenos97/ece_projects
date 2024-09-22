`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   12:32:51 01/06/2022
// Design Name:   DECODE
// Module Name:   /home/ise/Desktop/part1/decode_tb.v
// Project Name:  part1
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: DECODE
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module decode_tb;

	// Inputs
	reg [31:0] Instr;
	reg [31:0] ALU_out;
	reg [31:0] MEM_out;
	reg RF_WrEn;
	reg RF_WrData_sel;
	reg RF_B_sel;
	reg Clk;

	// Outputs
	wire [31:0] RF_A;
	wire [31:0] RF_B;
	wire [31:0] Immed;

	// Instantiate the Unit Under Test (UUT)
	DECODE uut (
		.RF_A(RF_A), 
		.RF_B(RF_B), 
		.Immed(Immed), 
		.Instr(Instr), 
		.ALU_out(ALU_out), 
		.MEM_out(MEM_out), 
		.RF_WrEn(RF_WrEn), 
		.RF_WrData_sel(RF_WrData_sel), 
		.RF_B_sel(RF_B_sel), 
		.Clk(Clk)
	);

	initial begin
		// Initialize Inputs
		ALU_out = 32'b1;
		MEM_out = 32'b10;
		RF_B_sel = 1;
		Clk = 0;
		
		// first set RF_WrEn to 1 to initialize register 1 and 2 contents during the first two clock cycles - data are coming 
		// both from the Alu_out and Mem_out inputs by setting RF_WrData_sel to 0 and 1 respectively
		RF_WrEn = 1;
		
		// --------------------------------------------------------------------------------------------------------------
		// dummy instruction written to trigger the following actions: with the RF_WrEn set to 1 decode unit should write 
		// register 1 (bits 20:16) and also the Immed16to32 subUnit should perform a sign extent
		Instr = 32'b11100000001000011000000000000010;
		
		// fetch data to write from the MEM_out
		RF_WrData_sel = 1; 
		
		// --------------------------------------------------------------------------------------------------------------
		#2 // delay until the next clock falling edge (clock cycle is 2ps and clock starts at 0)
		// dummy instruction written to trigger the following actions: with the RF_WrEn set to 1 the decode unit 
		// should write register 2 (bits 20:16) and also the Immed16to32 subUnit should perform a sign extent
		Instr = 32'b11100000001000101000000000000010;
		
		// fetch data to write from the ALU_out
		RF_WrData_sel = 0; 
		
		// --------------------------------------------------------------------------------------------------------------
		#1.5
		// dummy instruction written to trigger the following actions: with the RF_WrEn set to 0 and RF_B_Sel set to 1 the 
		// decode unit should read registers 1 and 2 (bits 25:21 and 20:16 respectively) inti RFA and RFB respectively and 
		// also the Immed16to32 subUnit should perform a sign extent
		Instr = 32'b11100000001000101000000000000010;
		
		// set RF_WrEn to 0 to allow for decode unit to read from the RF file
		RF_WrEn = 0;
		
	end
      
		// clock assignment always block
		always 
			begin
			#1 Clk = ~Clk;
		end
endmodule

