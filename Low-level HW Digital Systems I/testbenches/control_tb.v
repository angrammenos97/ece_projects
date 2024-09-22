`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   03:35:28 01/07/2022
// Design Name:   CONTROL
// Module Name:   /home/ise/Desktop/part1/control_tb.v
// Project Name:  part1
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: CONTROL
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module control_tb;

	// Inputs
	reg [31:0] Instr;
	reg Clk, Reset, ALU_zero;

	// Outputs
	//wire [2:0] currentState;
	wire [3:0] ALU_func;
	wire PC_LdEn, PC_sel, RF_B_sel, RF_WrData_sel,
	     RF_WrEn, ALU_Bin_sel, MEM_we, MEM_In_sel, MEM_Out_sel;

	// Instantiate the Unit Under Test (UUT)
	CONTROL uut ( 
		//.currentState(currentState), 
		.RF_WrData_sel(RF_WrData_sel),
		.Instr(Instr), 
		.Clk(Clk), 
		.Reset(Reset),
		.PC_LdEn(Pc_LdEn),
		.ALU_zero(ALU_zero),
		.PC_sel(PC_sel),
		.RF_B_sel(RF_B_sel),
		.RF_WrEn(Rf_Wr_En),
		.ALU_Bin_sel(ALU_Bin_sel),
		.ALU_func(ALU_func),
		.MEM_we(MEM_we),
		.MEM_In_sel(MEM_In_sel),
		.MEM_Out_sel(MEM_Out_sel)
	);


	initial 
		begin
		   // *********************************************signal initialization********************************************
			// initialize clock
			Clk = 1; // start clock from 1 to allow for reset signal to take effect                                    
			
			// Reset state and PC
			Reset      = 0;
			#0.5 Reset = 1;
			#0.5 Reset = 0;
			
			
			// *************************************************'b' testing***************************************************
			//Instr    = 32'b11111100000000000000000000000000; // initial command's opcode is set to test 'b' command's datapath and signals 
			                                                 // - expected PC_sel output is active
			//ALU_zero = 1'b1;                                 // set ALU_zero initially to test 'be' command
			
			
			
			// *************************************************'beq' testing***************************************************
			// after 2 clock cycles that the FSM is again in initial state test command 'beq' so set proper opcode and ALU_zero = 1
			#7 // account for initial state of clock being '1'
			Instr    = 32'b00000000000000000000000000000000;
			ALU_zero = 1'b1;                                 // for this value expected PC_sel output is active
			
			// after 3 clock cycles that the FSM is again in initial state change ALU_zero to '0' - expected PC_sel output is '0'
			#12 
			ALU_zero = 1'b0;
			
			
			// *************************************************'bne' testing***************************************************
			// after 3 clock cycles that the FSM is again in initial state test command 'bne' so set proper opcode and ALU_zero = 0
			#12
			Instr    = 32'b00000100000000000000000000000000;
			ALU_zero = 1'b0;
			
			// after 3 clock cycles that the FSM is again in initial state change ALU_zero to '1' - expected PC_sel output is '1'
			#12 
			ALU_zero = 1'b1;
			
			// *************************************************'ALU_op' testing***************************************************
			// after 3 clock cycles change the instruction to an ALU_opertion opcode '100000' - expected PC_Sel should  be '0' for both 
			// ALU_zero states
			
			// for the three functions below the ALU should perform an addition, thus ALU_func == 4'b0000
			#12
			Instr    = 32'b10000000000000000000000000000000; // addition
			
			#8
			Instr    = 32'b00111100000000000000000000000000; // load word
			
			#12
			Instr    = 32'b00011100000000000000000000000000; // store byte
			
			// each instruction describes the expected function in the side comments
			#8
			Instr = 32'b10000000000000000000000000110011; // logical OR
			
			#8
			Instr = 32'b11001000000000000000000000110011; // logical AND
			
			#8
			Instr = 32'b11111100000000000000000000000000; // branch

         #8
         Instr = 32'b11001100000000000000000000110011; // logical OR			
			
			
			
		end
	
	// clock's period is 4ns, since the polarity changes every 2ns (half period)
	always 
		begin
			#2 Clk = ~Clk;
		end
      
endmodule

