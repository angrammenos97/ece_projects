`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    22:03:32 01/11/2022 
// Design Name: 
// Module Name:    Proccessor 
// Project Name: 
// Target Devices: 
// Tool versions: 
// Description: 
//
// Dependencies: 
//
// Revision: 
// Revision 0.01 - File Created
// Additional Comments: 
//
//////////////////////////////////////////////////////////////////////////////////
module Proccessor(
   output wire [31:0] RF_A_buf, RF_B_buf,
	input wire Clk, Reset
    );

   //************************ MEM signals and modules ************************//
	wire [31:0] MEM_DataIn_muxed, MEM_DataOut, RF_B_ZeroFilled, 
	            MEM_DataOut_muxed, MEM_DataOut_ZeroFilled;
   wire WrEn,  MEM_In_sel, MEM_Out_sel, MEM_we;

   
	
	//****************************** ALU signals ******************************//
	wire [31:0] RF_A, RF_B, Immed, ALU_out;
	wire ALU_Bin_sel, ALU_Zero;
	wire [3:0] ALU_func; 
	
	
	//**************************** DECODE signals *****************************//
	wire [31:0] Instr;
	wire RF_WrData_sel, RF_B_sel, RF_WrEn;
	
	
	//****************************** PC signals *******************************//
	wire PC_sel, PC_LdEn;	
	
	
	//************************** module instantiation**************************//
	CONTROL controlUnit(	.PC_LdEn(PC_LdEn), .PC_sel(PC_sel), .RF_B_sel(RF_B_sel), .RF_WrEn(RF_WrEn), 
								.RF_WrData_sel(RF_WrData_sel), .ALU_Bin_sel(ALU_Bin_sel), .MEM_we(MEM_we),
								.MEM_In_sel(MEM_In_sel), .MEM_Out_sel(MEM_Out_sel), .ALU_func(ALU_func), 
								.Instr(Instr), .Clk(Clk), .Reset(Reset), .ALU_zero(ALU_Zero));
								
	IFSTAGE ifUnit(.PC_Immed(Immed), .PC_sel(PC_sel), .PC_LdEn(PC_LdEn), .Reset(Reset), .Clk(Clk), .Instr(Instr));
	
	DECODE decodeUnit(.Instr(Instr), .RF_WrEn(RF_WrEn), .ALU_out(ALU_out), .MEM_out(MEM_DataOut_muxed), 
							.RF_WrData_sel(RF_WrData_sel), .RF_B_sel(RF_B_sel), .Clk(Clk), .Immed(Immed), .RF_A(RF_A), .RF_B(RF_B));
	
	ALUSTAGE aluUnit(.RF_A(RF_A), .RF_B(RF_B), .Immed(Immed), .ALU_Bin_sel(ALU_Bin_sel), .ALU_func(ALU_func), .ALU_out(ALU_out), .ALU_Zero(ALU_Zero));
	
	MEMSTAGE ramUnit(.clk(Clk), .we(MEM_we), .addr(ALU_out[11:2]), .din(MEM_DataIn_muxed), .dout(MEM_DataOut));
	
	// multiplexer used to select between RF_B and ZeroFilled RF_B in MEM_In
	Mux2to1 MEM_In_mux21(.Out(MEM_DataIn_muxed), .In({RF_B, RF_B_ZeroFilled}), .Sel(MEM_In_sel));	
	
	// multiplexer used to select between MEM_out and ZeroFilled MEM_Out
	Mux2to1 MEM_Out_mux21(.Out(MEM_DataOut_muxed), .In({MEM_DataOut, MEM_DataOut_ZeroFilled}), .Sel(MEM_Out_sel));
	
	
	
	//*************************** signal assignment **************************//
	assign MEM_DataOut_ZeroFilled = {24'b0, MEM_DataOut[7:0]};
   assign RF_B_ZeroFilled        = {24'b0, RF_B[7:0]};	
	 
	// assign output to be able to synthesize module
   assign RF_A_buf = RF_A;
	assign RF_B_buf = RF_B;  
  

endmodule
